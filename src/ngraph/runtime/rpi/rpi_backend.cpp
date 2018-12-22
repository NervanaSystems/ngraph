//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/rpi/rpi_backend.hpp"
#include <omp.h>
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/any_all_replacement.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/util.hpp"
using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::rpi::RPIBackend();
}

runtime::rpi::RPIBackend::RPIBackend()
{
}

runtime::rpi::RPIBackend::RPIBackend(const vector<string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()}
{
}

shared_ptr<runtime::Tensor> runtime::rpi::RPIBackend::create_tensor(const element::Type& type,
                                                                    const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape, this);
}

shared_ptr<runtime::Tensor> runtime::rpi::RPIBackend::create_tensor(const element::Type& type,
                                                                    const Shape& shape,
                                                                    void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer, this);
}

runtime::Handle runtime::rpi::RPIBackend::compile(shared_ptr<Function> function)
{
    FunctionInstance& instance = m_function_map[function];
    if (!instance.m_is_compiled)
    {
        instance.m_is_compiled = true;
        pass::Manager pass_manager;
        // pass_manager.register_pass<pass::AnyAllReplacement>();
        pass_manager.register_pass<pass::LikeReplacement>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.register_pass<pass::ZeroDimTensorElimination>();
        pass_manager.register_pass<pass::AlgebraicSimplification>();
        // pass_manager.register_pass<pass::CoreFusion>();
        // pass_manager.register_pass<pass::ConstantFolding>();
        pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
        // pass_manager.register_pass<pass::GetOutputElementElimination>();
        pass_manager.register_pass<pass::Liveness>();
        pass_manager.register_pass<pass::MemoryLayout>(get_alignment());
        pass_manager.run_passes(function);

        size_t memory_pool_size = function->get_temporary_pool_size();
        instance.m_temporary_memory.reset(new AlignedBuffer(memory_pool_size, get_alignment()));

        for (const shared_ptr<Node>& node : function->get_ordered_ops())
        {
            instance.m_wrapped_nodes.emplace_back(node);
        }
    }

    return function;
}

bool runtime::rpi::RPIBackend::call(shared_ptr<Function> function,
                                    const vector<shared_ptr<runtime::Tensor>>& outputs,
                                    const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    auto fit = m_function_map.find(function);
    if (fit == m_function_map.end())
    {
        throw runtime_error("compile() must be called before call().");
    }
    FunctionInstance& instance = fit->second;
    if (!instance.m_is_compiled)
    {
        throw runtime_error("compile() must be called before call().");
    }

    // convert inputs to HostTensor
    vector<void*> func_inputs;
    vector<shared_ptr<runtime::HostTensor>> htv_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(static_cast<void*>(host_tensor->get_data_ptr()));
        htv_inputs.push_back(host_tensor);
    }

    // convert outputs to HostTensor
    vector<void*> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(static_cast<void*>(host_tensor->get_data_ptr()));
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, void*> tensor_map;
    size_t input_count = 0;
    for (auto param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = param->get_output_tensor_ptr(i).get();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < function->get_output_size(); ++output_count)
    {
        auto output = function->get_output_op(output_count);
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = output->get_output_tensor_ptr(0).get();
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (const NodeWrapper& wrapped : instance.m_wrapped_nodes)
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
        vector<shared_ptr<runtime::HostTensor>> htv_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = op->get_output_tensor_ptr(i).get();
            void* host_tensor = nullptr;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                auto offset = op->get_output_tensor(i).get_pool_offset();
                host_tensor = instance.get_temporary_pointer(offset);
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
            htv_outputs.push_back(make_shared<runtime::HostTensor>(
                tensor->get_element_type(), tensor->get_shape(), host_tensor, this));
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

        if (instance.m_performance_counters_enabled)
        {
            instance.m_timer_map[op].start();
        }
        generate_calls(type, wrapped, op_outputs, op_inputs, instance);
        if (instance.m_performance_counters_enabled)
        {
            instance.m_timer_map[op].stop();
        }
    }

    return true;
}

void runtime::rpi::RPIBackend::generate_calls(const element::Type& type,
                                              const NodeWrapper& op,
                                              const vector<void*>& outputs,
                                              const vector<const void*>& inputs,
                                              FunctionInstance& instance)
{
    stringstream ss;
    switch (type.get_type_enum())
    {
    case element::Type_t::boolean: op_engine<char>(op, outputs, inputs, instance); break;
    case element::Type_t::f32: op_engine<float>(op, outputs, inputs, instance); break;
    case element::Type_t::f64: op_engine<double>(op, outputs, inputs, instance); break;
    case element::Type_t::i8: op_engine<int8_t>(op, outputs, inputs, instance); break;
    case element::Type_t::i16: op_engine<int16_t>(op, outputs, inputs, instance); break;
    case element::Type_t::i32: op_engine<int32_t>(op, outputs, inputs, instance); break;
    case element::Type_t::i64: op_engine<int64_t>(op, outputs, inputs, instance); break;
    case element::Type_t::u8: op_engine<uint8_t>(op, outputs, inputs, instance); break;
    case element::Type_t::u16: op_engine<uint16_t>(op, outputs, inputs, instance); break;
    case element::Type_t::u32: op_engine<uint32_t>(op, outputs, inputs, instance); break;
    case element::Type_t::u64: op_engine<uint64_t>(op, outputs, inputs, instance); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::bf16:
        ss << "unsupported element type " << type << " op " << op.get_node().get_name();
        throw ngraph_error(ss.str());
    }
}

void runtime::rpi::RPIBackend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::rpi::RPIBackend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_map.at(func);
    for (const pair<const Node*, stopwatch> p : instance.m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}

bool runtime::rpi::RPIBackend::is_supported(const Node& node) const
{
    return m_unsupported_op_name_list.find(node.description()) == m_unsupported_op_name_list.end();
}
