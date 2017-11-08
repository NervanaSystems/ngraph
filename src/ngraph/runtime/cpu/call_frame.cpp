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
#include "ngraph/runtime/cpu/tensor_view.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

CallFrame::CallFrame(EntryPoint compiled_function,
                     const std::vector<std::shared_ptr<CallFrame>>& callees)
    : m_compiled_function(compiled_function)
    , m_callees(callees)
{
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& input_tvs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& output_tvs)
{
    m_inputs.clear();
    m_outputs.clear();
    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(input_tvs[i]);
        m_inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(output_tvs[i]);
        m_outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    m_compiled_function(this);
}

void CallFrame::operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& arguments,
                           const std::vector<std::shared_ptr<ngraph::runtime::Value>>& results)
{
    // TODO: Check types of args and result
    vector<shared_ptr<ngraph::runtime::TensorView>> inputs;
    for (shared_ptr<ngraph::runtime::Value> argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<ngraph::runtime::TensorView>> outputs;
    for (shared_ptr<ngraph::runtime::Value> result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}

void* CallFrame::get_input_data(size_t index)
{
    void* rc = m_inputs.at(index);
    return rc;
}

void* CallFrame::get_output_data(size_t index)
{
    void* rc = m_outputs.at(index);
    return rc;
}
