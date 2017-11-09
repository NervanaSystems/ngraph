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
    // NGRAPH_INFO << "tensor_call";
    // for (shared_ptr<ngraph::runtime::TensorView> input : inputs)
    // {
    //     NGRAPH_INFO << input->get_tensor_view_descriptor()->get_name();
    // }

    // for (shared_ptr<ngraph::runtime::TensorView> output : outputs)
    // {
    //     NGRAPH_INFO << output->get_tensor_view_descriptor()->get_name();
    // }

    // Invoke compiled computation
    m_compiled_function(this, inputs, outputs);
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
