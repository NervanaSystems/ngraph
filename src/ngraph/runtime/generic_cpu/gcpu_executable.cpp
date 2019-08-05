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

#include "ngraph/runtime/generic_cpu/gcpu_executable.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::gcpu::GCPUExecutable::GCPUExecutable(const shared_ptr<Function>& function,
                                              bool enable_performance_collection)
    : m_is_compiled{true}
    , m_performance_counters_enabled{enable_performance_collection}
{
    m_function = clone_function(*function);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::FusedOpDecomposition>();
    pass_manager.register_pass<pass::ImplicitBroadcastElimination>();
    pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(m_function);

    for (const shared_ptr<Node>& node : m_function->get_ordered_ops())
    {
        m_wrapped_nodes.emplace_back(node);
    }
    set_parameters_and_results(*m_function);
}

runtime::gcpu::GCPUExecutable::GCPUExecutable(const std::string& model_string)
    : m_is_compiled{true}
    , m_performance_counters_enabled{false}
{
    m_function = deserialize(model_string);
    for (const shared_ptr<Node>& node : m_function->get_ordered_ops())
    {
        m_wrapped_nodes.emplace_back(node);
    }
    set_parameters_and_results(*m_function);
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
    if (m_nan_check_enabled)
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
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = &output->output(0).get_tensor();
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (const NodeWrapper& wrapped : m_wrapped_nodes)
    {
        auto op = wrapped.get_node();
        auto type_id = wrapped.get_typeid();
        if (type_id == OP_TYPEID::Parameter)
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
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        if (m_performance_counters_enabled)
        {
            m_timer_map[op].start();
        }
        generate_calls(type, wrapped, op_outputs, op_inputs);
        if (m_performance_counters_enabled)
        {
            m_timer_map[op].stop();
        }
        if (m_nan_check_enabled)
        {
            perform_nan_check(op_outputs, op.get());
        }
    }

    return true;
}

void runtime::gcpu::GCPUExecutable::generate_calls(const element::Type& type,
                                                   const NodeWrapper& op,
                                                   const vector<shared_ptr<HostTensor>>& out,
                                                   const vector<shared_ptr<HostTensor>>& in)
{
    stringstream ss;
    switch (type.get_type_enum())
    {
    case element::Type_t::boolean: op_engine<char>(op, out, in); break;
    case element::Type_t::f32: op_engine<float>(op, out, in); break;
    case element::Type_t::f64: op_engine<double>(op, out, in); break;
    case element::Type_t::i8: op_engine<int8_t>(op, out, in); break;
    case element::Type_t::i16: op_engine<int16_t>(op, out, in); break;
    case element::Type_t::i32: op_engine<int32_t>(op, out, in); break;
    case element::Type_t::i64: op_engine<int64_t>(op, out, in); break;
    case element::Type_t::u8: op_engine<uint8_t>(op, out, in); break;
    case element::Type_t::u16: op_engine<uint16_t>(op, out, in); break;
    case element::Type_t::u32: op_engine<uint32_t>(op, out, in); break;
    case element::Type_t::u64: op_engine<uint64_t>(op, out, in); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::bf16:
    case element::Type_t::f16:
        ss << "unsupported element type " << type << " op " << op.get_node()->get_name();
        throw ngraph_error(ss.str());
    }
}

void runtime::gcpu::GCPUExecutable::set_nan_check(bool enable)
{
    m_nan_check_enabled = enable;
}

vector<runtime::PerformanceCounter> runtime::gcpu::GCPUExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (const pair<shared_ptr<const Node>, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first, p.second.get_total_microseconds(), p.second.get_call_count());
    }
    return rc;
}

void runtime::gcpu::GCPUExecutable::perform_nan_check(const vector<shared_ptr<HostTensor>>& tensors,
                                                      const Node* op)
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

void runtime::gcpu::GCPUExecutable::save(ostream& out)
{
    cpio::Writer writer(out);
    string si = "INTERPRETER Save File 1.0";
    writer.write("save_info", si.data(), si.size());
    string model = serialize(m_function, 0);
    writer.write("model", model.data(), model.size());
}
