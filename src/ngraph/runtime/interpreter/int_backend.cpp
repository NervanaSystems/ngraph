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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/backend_manager.hpp"
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
    return new runtime::interpreter::INTBackend();
}

runtime::interpreter::INTBackend::INTBackend()
{
}

runtime::interpreter::INTBackend::INTBackend(const vector<string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()}
{
}

shared_ptr<runtime::Tensor>
    runtime::interpreter::INTBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape, this);
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer, this);
}

runtime::Handle runtime::interpreter::INTBackend::compile(shared_ptr<Function> function)
{
    FunctionInstance& instance = m_function_map[function];
    if (!instance.m_is_compiled)
    {
        instance.m_is_compiled = true;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::LikeReplacement>();
        pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
        pass_manager.register_pass<pass::Liveness>();
        pass_manager.run_passes(function);

        for (const shared_ptr<Node>& node : function->get_ordered_ops())
        {
            instance.m_wrapped_nodes.emplace_back(node);
        }
    }

    return function;
}

bool runtime::interpreter::INTBackend::call(shared_ptr<Function> function,
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
    vector<shared_ptr<HostTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }
    if (instance.m_nan_check_enabled)
    {
        perform_nan_check(func_inputs);
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

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (const descriptor::Input& input : op->get_inputs())
        {
            descriptor::Tensor* tensor = input.get_output().get_tensor_ptr().get();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = op->get_output_tensor_ptr(i).get();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                const Shape& shape = op->get_output_shape(i);
                const element::Type& type = op->get_output_element_type(i);
                string name = op->get_output_tensor(i).get_name();
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
        if (instance.m_nan_check_enabled)
        {
            perform_nan_check(op_outputs, op);
        }
    }

    return true;
}

void runtime::interpreter::INTBackend::generate_calls(const element::Type& type,
                                                      const NodeWrapper& op,
                                                      const vector<shared_ptr<HostTensor>>& outputs,
                                                      const vector<shared_ptr<HostTensor>>& inputs,
                                                      FunctionInstance& instance)
{
    vector<void*> out;
    vector<const void*> in;
    for (auto t : outputs)
    {
        out.push_back(t->get_data_ptr());
    }
    for (auto t : inputs)
    {
        in.push_back(t->get_data_ptr());
    }
    stringstream ss;
    switch (type.get_type_enum())
    {
    case element::Type_t::boolean: op_engine<char>(op, out, in, instance); break;
    case element::Type_t::f32: op_engine<float>(op, out, in, instance); break;
    case element::Type_t::f64: op_engine<double>(op, out, in, instance); break;
    case element::Type_t::i8: op_engine<int8_t>(op, out, in, instance); break;
    case element::Type_t::i16: op_engine<int16_t>(op, out, in, instance); break;
    case element::Type_t::i32: op_engine<int32_t>(op, out, in, instance); break;
    case element::Type_t::i64: op_engine<int64_t>(op, out, in, instance); break;
    case element::Type_t::u8: op_engine<uint8_t>(op, out, in, instance); break;
    case element::Type_t::u16: op_engine<uint16_t>(op, out, in, instance); break;
    case element::Type_t::u32: op_engine<uint32_t>(op, out, in, instance); break;
    case element::Type_t::u64: op_engine<uint64_t>(op, out, in, instance); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::bf16:
        ss << "unsupported element type " << type << " op " << op.get_node().get_name();
        throw ngraph_error(ss.str());
    }
}

void runtime::interpreter::INTBackend::set_nan_check(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_nan_check_enabled = enable;
}

void runtime::interpreter::INTBackend::enable_performance_data(shared_ptr<Function> func,
                                                               bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INTBackend::get_performance_data(shared_ptr<Function> func) const
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

void runtime::interpreter::INTBackend::perform_nan_check(
    const vector<shared_ptr<HostTensor>>& tensors, const Node* op)
{
    size_t arg_number = 1;
    for (const shared_ptr<HostTensor>& tensor : tensors)
    {
        const element::Type& type = tensor->get_element_type();
        if (type == element::f32)
        {
            const float* data = tensor->get_data_ptr<float>();
            for (size_t i = 0; i < tensor->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        else if (type == element::f64)
        {
            const double* data = tensor->get_data_ptr<double>();
            for (size_t i = 0; i < tensor->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        arg_number++;
    }
}

bool runtime::interpreter::INTBackend::is_supported(const Node& node) const
{
    return m_unsupported_op_name_list.find(node.description()) == m_unsupported_op_name_list.end();
}
