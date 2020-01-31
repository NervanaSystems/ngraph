//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/gcpu/gcpu_executable.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::gcpu::GCPUExecutable::GCPUExecutable(const shared_ptr<Function>& function,
                                              bool enable_performance_collection)
    : INTExecutable(function, enable_performance_collection)
{
}

bool runtime::gcpu::GCPUExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                         const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // convert inputs to HostTensor
    vector<shared_ptr<HostTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }

    // convert outputs to HostTensor
    vector<shared_ptr<HostTensor>> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count)
    {
        auto output = get_results()[output_count];
        if (!is_type<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = &output->output(0).get_tensor();
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (auto& op : m_nodes)
    {
        auto type_id = get_typeid(*op);
        if (type_id == ngraph::runtime::interpreter::OP_TYPEID::Parameter)
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (auto input : op->inputs())
        {
            descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &op->output(i).get_tensor();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                const Shape& shape = op->get_output_shape(i);
                const element::Type& type = op->get_output_element_type(i);
                string name = op->output(i).get_tensor().get_name();
                host_tensor = make_shared<runtime::HostTensor>(type, shape, name);
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        // get op type
        element::Type type;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
        switch (type_id)
        {
        case ngraph::runtime::interpreter::OP_TYPEID::Convert:
        case ngraph::runtime::interpreter::OP_TYPEID::Quantize:
        case ngraph::runtime::interpreter::OP_TYPEID::Dequantize:
        case ngraph::runtime::interpreter::OP_TYPEID::ArgMin:
        case ngraph::runtime::interpreter::OP_TYPEID::ArgMax:
            type = op->get_input_element_type(0);
            break;
        case ngraph::runtime::interpreter::OP_TYPEID::Equal:
        case ngraph::runtime::interpreter::OP_TYPEID::Greater:
        case ngraph::runtime::interpreter::OP_TYPEID::GreaterEq:
        case ngraph::runtime::interpreter::OP_TYPEID::Less:
        case ngraph::runtime::interpreter::OP_TYPEID::LessEq:
        case ngraph::runtime::interpreter::OP_TYPEID::NotEqual:
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
            break;
        case ngraph::runtime::interpreter::OP_TYPEID::TopK:
            type = op->get_output_element_type(1);
            break;
        default: type = op->get_output_element_type(0); break;
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        if (m_performance_counters_enabled)
        {
            m_timer_map[op].start();
        }
        generate_calls(type, *op, op_outputs, op_inputs);
        if (m_performance_counters_enabled)
        {
            m_timer_map[op].stop();
        }
    }

    return true;
}

void runtime::gcpu::GCPUExecutable::generate_calls(const element::Type& type,
                                                   const Node& op,
                                                   const vector<shared_ptr<HostTensor>>& out,
                                                   const vector<shared_ptr<HostTensor>>& in)
{
    stringstream ss;
    switch (type)
    {
    case element::Type_t::boolean: gop_engine<char>(op, out, in); break;
    case element::Type_t::f32: gop_engine<float>(op, out, in); break;
    case element::Type_t::f64: gop_engine<double>(op, out, in); break;
    case element::Type_t::i8: gop_engine<int8_t>(op, out, in); break;
    case element::Type_t::i16: gop_engine<int16_t>(op, out, in); break;
    case element::Type_t::i32: gop_engine<int32_t>(op, out, in); break;
    case element::Type_t::i64: gop_engine<int64_t>(op, out, in); break;
    case element::Type_t::u8: gop_engine<uint8_t>(op, out, in); break;
    case element::Type_t::u16: gop_engine<uint16_t>(op, out, in); break;
    case element::Type_t::u32: gop_engine<uint32_t>(op, out, in); break;
    case element::Type_t::u64: gop_engine<uint64_t>(op, out, in); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::u1:
    case element::Type_t::bf16:
    case element::Type_t::f16:
        ss << "unsupported element type " << type << " op " << op.get_name();
        throw ngraph_error(ss.str());
    }
}
