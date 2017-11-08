// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>

#include "call_frame.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

CallFrame::CallFrame(EntryPoint compiled_function,
                     size_t n_outputs,
                     size_t n_inputs,
                     const TensorViewPtrs& temps,
                     const std::vector<std::shared_ptr<CallFrame>>& callees)
    : m_n_outputs(n_outputs)
    , m_n_inputs(n_inputs)
    , m_tensor_views(n_outputs + n_inputs + temps.size())
    , m_compiled_function(compiled_function)
    , m_callees(callees)
{
    copy(temps.begin(), temps.end(), m_tensor_views.begin() + m_n_outputs + m_n_inputs);
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outputs)
{
    NGRAPH_INFO << "tensor_call";
    for (shared_ptr<ngraph::runtime::TensorView> input : inputs)
    {
        NGRAPH_INFO << input->get_tensor_view_descriptor()->get_name();
    }

    for (shared_ptr<ngraph::runtime::TensorView> output : outputs)
    {
        NGRAPH_INFO << output->get_tensor_view_descriptor()->get_name();
    }

    copy(outputs.begin(), outputs.end(), m_tensor_views.begin());
    copy(inputs.begin(), inputs.end(), m_tensor_views.begin() + m_n_outputs);

    // Invoke compiled computation
    m_compiled_function(this, inputs, outputs, m_tensor_views, m_callees);

    // Don't hold onto inputs/outputs
    fill_n(m_tensor_views.begin(), m_n_outputs + m_n_inputs, nullptr);
}

void CallFrame::operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& arguments,
                           const std::vector<std::shared_ptr<ngraph::runtime::Value>>& results)
{
    // TODO: Check types of args and result
    vector<shared_ptr<ngraph::runtime::TensorView>> inputs;
    for (shared_ptr<ngraph::runtime::Value> argument : arguments)
    {
        shared_ptr<runtime::TensorView> tv = dynamic_pointer_cast<runtime::TensorView>(argument);
        NGRAPH_INFO << tv->get_tensor_view_descriptor()->get_name();
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<ngraph::runtime::TensorView>> outputs;
    for (shared_ptr<ngraph::runtime::Value> result : results)
    {
        // NGRAPH_INFO << result->get_tensor_view_descriptor()->get_name();
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}
