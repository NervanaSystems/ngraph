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
#include "ngraph/runtime/chrome_trace.hpp"
#include "ngraph/runtime/interpreter/int_tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::interpreter::INTExecutable::INTExecutable(const shared_ptr<Function>& function,
                                                   bool enable_performance_collection)
    : m_is_compiled{true}
    , m_performance_counters_enabled{enable_performance_collection}
{
    m_function = clone_function(*function);

    for (const shared_ptr<Node>& node : m_function->get_ordered_ops())
    {
        m_wrapped_nodes.emplace_back(node);
    }
    set_parameters_and_results(*m_function);
}

runtime::interpreter::INTExecutable::INTExecutable(const std::string& model_string)
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

bool runtime::interpreter::INTExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                               const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // convert inputs to INTTensor
    vector<shared_ptr<INTTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<INTTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }

    // convert outputs to INTTensor
    vector<shared_ptr<INTTensor>> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<INTTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> INTTensor
    unordered_map<descriptor::Tensor*, shared_ptr<INTTensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> INTTensor
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
        vector<shared_ptr<INTTensor>> op_inputs;
        for (auto input : op->inputs())
        {
            descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<INTTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &op->output(i).get_tensor();
            shared_ptr<INTTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                const Shape& shape = op->get_output_shape(i);
                const element::Type& type = op->get_output_element_type(i);
                string name = op->output(i).get_tensor().get_name();
                host_tensor = make_shared<INTTensor>(type, shape, name);
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        if (m_performance_counters_enabled)
        {
            m_timer_map[op].start();
        }
        execute_op(wrapped, op_outputs, op_inputs);
        if (m_performance_counters_enabled)
        {
            m_timer_map[op].stop();
        }
    }

    return true;
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INTExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (const pair<shared_ptr<const Node>, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first, p.second.get_total_microseconds(), p.second.get_call_count());
    }
    return rc;
}

void runtime::interpreter::INTExecutable::save(ostream& out)
{
    cpio::Writer writer(out);
    string si = "INTERPRETER Save File 1.0";
    writer.write("save_info", si.data(), si.size());
    string model = serialize(m_function, 0);
    writer.write("model", model.data(), model.size());
}

shared_ptr<ngraph::op::Parameter>
    runtime::interpreter::INTExecutable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::interpreter::INTExecutable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor>
    runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<INTTensor>(parameter->get_element_type(), parameter->get_shape());
}

shared_ptr<runtime::Tensor>
    runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<INTTensor>(result->get_element_type(), result->get_shape());
}

vector<shared_ptr<runtime::Tensor>>
    runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index,
                                                             size_t pipeline_depth)
{
    vector<shared_ptr<INTTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<INTTensor> tensor;
        auto t = make_shared<INTTensor>(parameter->get_element_type(), parameter->get_shape());
        tensor = static_pointer_cast<INTTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<INTTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>>
    runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index,
                                                              size_t pipeline_depth)
{
    vector<shared_ptr<INTTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<INTTensor> tensor;
        auto t = make_shared<INTTensor>(result->get_element_type(), result->get_shape());
        tensor = static_pointer_cast<INTTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<INTTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

void runtime::interpreter::INTExecutable::execute_op(const NodeWrapper& op,
                                                     const vector<shared_ptr<INTTensor>>& outputs,
                                                     const vector<shared_ptr<INTTensor>>& inputs)
{
    switch (op.get_typeid())
    {
    case OP_TYPEID::Abs:
        runtime::reference::abs(inputs[0]->get_value(), outputs[0]->get_value());
        break;
    case OP_TYPEID::Add:
        runtime::reference::add(
            inputs[0]->get_value(), inputs[1]->get_value(), outputs[0]->get_value());
        break;
    case OP_TYPEID::Result:
        reference::result(inputs[0]->get_value(), outputs[0]->get_value());
        break;
    }
}
