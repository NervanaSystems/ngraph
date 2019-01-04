//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/runtime/interpreter/int_executable.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"

using namespace std;
using namespace ngraph;

runtime::interpreter::INTExecutable::INTExecutable(
                                                   shared_ptr<Function> function,
                                                   bool enable_performance_collection)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::MemoryLayout>(get_alignment());
    pass_manager.run_passes(function);

    size_t memory_pool_size = function->get_temporary_pool_size();
    m_temporary_memory.reset(new AlignedBuffer(memory_pool_size, get_alignment()));

    for (const shared_ptr<Node>& node : function->get_ordered_ops())
    {
        m_wrapped_nodes.emplace_back(node);
    }

    set_parameters_and_results(*function);
}

bool runtime::interpreter::INTExecutable::execute(const vector<runtime::Tensor*>& outputs,
                                                  const vector<runtime::Tensor*>& inputs)
{
    // convert inputs to HostTensor
    vector<void*> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_cast<runtime::HostTensor*>(tensor);
        func_inputs.push_back(static_cast<void*>(host_tensor->get_data_ptr()));
    }

    // convert outputs to HostTensor
    vector<void*> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_cast<runtime::HostTensor*>(tensor);
        func_outputs.push_back(static_cast<void*>(host_tensor->get_data_ptr()));
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, void*> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = param->get_output_tensor_ptr(i).get();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    size_t output_count = 0;
    for (auto output : get_results())
    {
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = output->get_output_tensor_ptr(0).get();
        tensor_map.insert({tensor, func_outputs[output_count++]});
    }

    // for each ordered op in the graph
    for (const NodeWrapper& wrapped : m_wrapped_nodes)
    {
        const Node* op = &wrapped.get_node();
        auto type_id = wrapped.get_typeid();
        if (type_id == OP_TYPEID::Parameter)
        {
            continue;
        }
        if (type_id == OP_TYPEID::Constant)
        {
            const op::Constant* c = static_cast<const op::Constant*>(op);
            descriptor::Tensor* tensor = op->get_output_tensor_ptr(0).get();
            tensor_map.insert({tensor, const_cast<void*>(c->get_data_ptr())});
            continue;
        }
        // get op inputs from map
        vector<const void*> op_inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::Tensor* tensor = input.get_output().get_tensor_ptr().get();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<void*> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = op->get_output_tensor_ptr(i).get();
            void* host_tensor = nullptr;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                auto offset = op->get_output_tensor(i).get_pool_offset();
                host_tensor = get_temporary_pointer(offset);
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"
        switch (type_id)
        {
        case OP_TYPEID::Convert:
        case OP_TYPEID::Quantize:
        case OP_TYPEID::Dequantize:
        case OP_TYPEID::ArgMin:
        case OP_TYPEID::ArgMax: type = op->get_input_element_type(0); break;
        case OP_TYPEID::Equal:
        case OP_TYPEID::Greater:
        case OP_TYPEID::GreaterEq:
        case OP_TYPEID::Less:
        case OP_TYPEID::LessEq:
        case OP_TYPEID::NotEqual:
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
            break;
        case OP_TYPEID::TopK: type = op->get_output_element_type(1); break;
        default: type = op->get_output_element_type(0); break;
        }
#pragma GCC diagnostic pop

        if (m_performance_counters_enabled)
        {
            m_timer_map[op].start();
        }
        generate_calls(type, wrapped, op_outputs, op_inputs);
        if (m_performance_counters_enabled)
        {
            m_timer_map[op].stop();
        }
    }

    return true;
}

void runtime::interpreter::INTExecutable::generate_calls(const element::Type& type,
                                                         const NodeWrapper& op,
                                                         const vector<void*>& outputs,
                                                         const vector<const void*>& inputs)
{
    stringstream ss;
    switch (type.get_type_enum())
    {
    case element::Type_t::boolean: op_engine<char>(op, outputs, inputs); break;
    case element::Type_t::f32: op_engine<float>(op, outputs, inputs); break;
    case element::Type_t::f64: op_engine<double>(op, outputs, inputs); break;
    case element::Type_t::i8: op_engine<int8_t>(op, outputs, inputs); break;
    case element::Type_t::i16: op_engine<int16_t>(op, outputs, inputs); break;
    case element::Type_t::i32: op_engine<int32_t>(op, outputs, inputs); break;
    case element::Type_t::i64: op_engine<int64_t>(op, outputs, inputs); break;
    case element::Type_t::u8: op_engine<uint8_t>(op, outputs, inputs); break;
    case element::Type_t::u16: op_engine<uint16_t>(op, outputs, inputs); break;
    case element::Type_t::u32: op_engine<uint32_t>(op, outputs, inputs); break;
    case element::Type_t::u64: op_engine<uint64_t>(op, outputs, inputs); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::bf16:
        ss << "unsupported element type " << type << " op " << op.get_node().get_name();
        throw ngraph_error(ss.str());
    }
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INTExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (const pair<const Node*, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}
