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
                     const TensorViewPtrs& temps)

    : m_n_outputs(n_outputs)
    , m_n_inputs(n_inputs)
    , m_tensor_views(n_inputs + n_outputs + temps.size())
    , m_compiled_function(compiled_function)
{
    copy(temps.begin(), temps.end(), m_tensor_views.begin() + m_n_outputs + m_n_inputs);
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outputs)
{
    copy(outputs.begin(), outputs.end(), m_tensor_views.begin());
    copy(inputs.begin(), inputs.end(), m_tensor_views.begin() + m_n_outputs);

    // Invoke compiled computation
    m_compiled_function(this, m_tensor_views);

    // Don't hold onto inputs/outputs
    fill_n(m_tensor_views.begin(), m_n_outputs + m_n_inputs, nullptr);
}

void CallFrame::operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& arguments,
                           const std::vector<std::shared_ptr<ngraph::runtime::Value>>& results)
{
    // TODO: Check types of args and result
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> inputs;
    for (auto argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> outputs;
    for (auto result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}
