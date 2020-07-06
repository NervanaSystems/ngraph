//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::FunctionCall::type_info;

op::FunctionCall::FunctionCall(const vector<Output<Node>>& outputs,
                               const vector<Output<Node>>& inputs,
                               const Function& function)
    : Op(inputs)
    , m_function_outputs{outputs}
    , m_function{ngraph::clone_function(function)}
{
    set_output_size(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        set_output_type(i, outputs[i].get_element_type(), outputs[i].get_partial_shape());
    }
}

const string& op::FunctionCall::description() const
{
    static string s_type = "FunctionCall";
    return s_type;
}

shared_ptr<Node> op::FunctionCall::clone_with_new_inputs(const OutputVector& new_args) const
{
    return make_shared<FunctionCall>(m_function_outputs, new_args, *m_function);
}

shared_ptr<runtime::Backend> op::FunctionCall::get_backend() const
{
    return m_backend;
}

shared_ptr<runtime::Executable> op::FunctionCall::get_executable() const
{
    return m_executable;
}

shared_ptr<Function> op::FunctionCall::get_function() const
{
    return m_function;
}
