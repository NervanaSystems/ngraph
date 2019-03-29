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

#include "function_call.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace std;
using namespace ngraph;

runtime::hybrid::op::FunctionCall::FunctionCall(const NodeVector& outputs,
                                                const NodeVector& inputs,
                                                shared_ptr<Function> function,
                                                shared_ptr<Backend> backend)
    : Op("FunctionCall", inputs)
    , m_function_outputs{outputs}
    , m_function{function}
    , m_backend{backend}
    , m_executable{backend->compile(function)}
{
    set_output_size(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        set_output_type(i, outputs[i]->get_element_type(), outputs[i]->get_output_shape(0));
    }
}

shared_ptr<Node>
    runtime::hybrid::op::FunctionCall::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<FunctionCall>(m_function_outputs, new_args, m_function, m_backend);
}

shared_ptr<runtime::Backend> runtime::hybrid::op::FunctionCall::get_backend() const
{
    return m_backend;
}

shared_ptr<runtime::Executable> runtime::hybrid::op::FunctionCall::get_executable() const
{
    return m_executable;
}

shared_ptr<Function> runtime::hybrid::op::FunctionCall::get_function() const
{
    return m_function;
}
