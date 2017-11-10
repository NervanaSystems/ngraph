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
                     const std::vector<std::shared_ptr<CallFrame>>& callees)
    : m_compiled_function(compiled_function)
    , m_callees(callees)
{
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outputs)
{
    // process inputs
    for (std::shared_ptr<ngraph::runtime::TensorView> tv : inputs)
    {
    }

    // process outputs
    for (std::shared_ptr<ngraph::runtime::TensorView> tv : outputs)
    {
    }

    // Invoke compiled computation
    // m_compiled_function(this, inputs, outputs);
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
    // shared_ptr<runtime::TensorView> p = m_inputs->at(index);
    // shared_ptr<const runtime::ParameterizedTensorView<element::Float32>> ptv =
    //     dynamic_pointer_cast<const runtime::ParameterizedTensorView<element::Float32>>(p);
    // const void* p1 = ptv->get_vector().data();
    // return const_cast<void*>(p1);
    NGRAPH_INFO << "fix this";
    return nullptr;
}

void* CallFrame::get_output_data(size_t index)
{
    // shared_ptr<runtime::TensorView> p = m_outputs->at(index);
    // shared_ptr<runtime::ParameterizedTensorView<element::Float32>> ptv =
    //     dynamic_pointer_cast<runtime::ParameterizedTensorView<element::Float32>>(p);
    // return ptv->get_vector().data();
    NGRAPH_INFO << "fix this";
    return nullptr;
}
