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

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Executable::Executable(const shared_ptr<runtime::Backend>& backend)
    : m_backend{backend}
{
}

runtime::Executable::~Executable()
{
}

shared_ptr<runtime::Tensor> runtime::Executable::create_input_tensor(size_t index,
                                                                     void* memory_pointer)
{
    shared_ptr<runtime::Tensor> tensor;
    if (m_backend)
    {
        const ParameterVector& parameters = get_parameters();
        if (index >= parameters.size())
        {
            throw runtime_error("create_tensor for input out of bounds");
        }
        shared_ptr<op::Parameter> parameter = parameters[index];
        tensor = m_backend->create_tensor(
            parameter->get_element_type(), parameter->get_shape(), memory_pointer);
    }
    else
    {
        throw runtime_error("Backend does not support Executable::create_tensor");
    }
    return tensor;
}

shared_ptr<runtime::Tensor> runtime::Executable::create_output_tensor(size_t index,
                                                                      void* memory_pointer)
{
    shared_ptr<runtime::Tensor> tensor;
    if (m_backend)
    {
        const ResultVector& results = get_results();
        if (index >= results.size())
        {
            throw runtime_error("create_tensor for input out of bounds");
        }
        shared_ptr<op::Result> result = results[index];
        tensor = m_backend->create_tensor(
            result->get_element_type(), result->get_shape(), memory_pointer);
    }
    else
    {
        throw runtime_error("Backend does not support Executable::create_tensor");
    }
    return tensor;
}

shared_ptr<runtime::Tensor>
    runtime::Executable::create_parameter_tensor(const op::Parameter& parameter)
{
    throw runtime_error("Unimplemented");
}

shared_ptr<runtime::Tensor> runtime::Executable::create_result_tensor(const Node& result)
{
    throw runtime_error("Unimplemented");
}

shared_ptr<runtime::Tensor>
    runtime::Executable::create_parameter_tensor(const shared_ptr<op::Parameter>& parameter)
{
    return create_parameter_tensor(*parameter);
}

shared_ptr<runtime::Tensor>
    runtime::Executable::create_result_tensor(const shared_ptr<Node>& result)
{
    return create_result_tensor(*result);
}

bool runtime::Executable::call_with_validate(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                             const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    validate(outputs, inputs);
    return call(outputs, inputs);
}

void runtime::Executable::validate(const vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                   const vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    const ParameterVector& parameters = get_parameters();
    const ResultVector& results = get_results();
    if (parameters.size() != inputs.size())
    {
        stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << parameters.size();
        throw runtime_error(ss.str());
    }
    if (results.size() != outputs.size())
    {
        stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count "
           << results.size();
        throw runtime_error(ss.str());
    }

    for (size_t i = 0; i < parameters.size(); i++)
    {
        if (parameters[i]->get_element_type().is_static() &&
            parameters[i]->get_element_type() != inputs[i]->get_element_type())
        {
            stringstream ss;
            ss << "Input " << i << " type '" << inputs[i]->get_element_type()
               << "' does not match Parameter type '" << parameters[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (!(parameters[i]->get_output_partial_shape(0).relaxes(inputs[i]->get_partial_shape())))
        {
            stringstream ss;
            ss << "Input " << i << " shape " << inputs[i]->get_partial_shape()
               << " does not match Parameter shape " << parameters[i]->get_output_partial_shape(0);
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < results.size(); i++)
    {
        if (outputs[i]->get_element_type().is_static() &&
            results[i]->get_element_type() != outputs[i]->get_element_type())
        {
            stringstream ss;
            ss << "Output " << i << " type '" << outputs[i]->get_element_type()
               << "' does not match Result type '" << results[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (!(outputs[i]->get_partial_shape()).relaxes(results[i]->get_output_partial_shape(0)))
        {
            stringstream ss;
            ss << "Output " << i << " shape " << outputs[i]->get_partial_shape()
               << " does not match Result shape " << results[i]->get_output_partial_shape(0);
            throw runtime_error(ss.str());
        }
    }
}

const ngraph::ParameterVector& runtime::Executable::get_parameters() const
{
    return m_parameters;
}

const ngraph::ResultVector& runtime::Executable::get_results() const
{
    return m_results;
}

void runtime::Executable::set_parameters_and_results(const Function& func)
{
    m_parameters = func.get_parameters();
    m_results = func.get_results();
}

vector<runtime::PerformanceCounter> runtime::Executable::get_performance_data() const
{
    return vector<PerformanceCounter>();
}

void runtime::Executable::save(std::ostream& output_stream)
{
    throw runtime_error("save opertion unimplemented.");
}
