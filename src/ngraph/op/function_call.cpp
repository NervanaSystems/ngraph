/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/function_call.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

op::FunctionCall::FunctionCall(shared_ptr<Function> function, const NodeVector& args)
    : Node("FunctionCall", args)
    , m_function(function)
{
    auto& function_params = m_function->get_parameters();

    // TODO : [nikolayk] this needs to be rewritten as follows
    // for each i : FunctionCall->get_inputs.at(i).get_tensor_view_type ==
    // flatten(function_parms).at(i)
    if (get_input_size() != function_params.size())
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (get_input_element_type(i) != function->get_parameters().at(i)->get_element_type() ||
            get_input_shape(i) != function->get_parameters().at(i)->get_shape())
        {
            throw ngraph_error("Function argument type mismatch.");
        }
    }

    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        add_output(function->get_output_element_type(i), function->get_output_shape(i));
    }
}

shared_ptr<Node> op::FunctionCall::copy_with_new_args(const NodeVector& new_args) const
{
    shared_ptr<FunctionCall> fc = make_shared<FunctionCall>(m_function, new_args);
    fc->m_function = clone_function(*m_function);
    return fc;
}

/// \return A singleton vector containing the function to be called.
vector<shared_ptr<Function>> op::FunctionCall::get_functions() const
{
    return vector<shared_ptr<Function>>{m_function};
}
