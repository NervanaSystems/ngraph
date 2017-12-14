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

#include "ngraph/ops/function_call.hpp"
#include "ngraph/function.hpp"
#include "ngraph/xla_function.hpp"

using namespace std;
using namespace ngraph;

op::FunctionCall::FunctionCall(std::shared_ptr<Function> function,
                               const std::vector<std::shared_ptr<Node>>& args)
    : Node("FunctionCall", args)
    , m_function(function)
{
    auto& function_params = m_function->get_parameters();

    //TODO : [nikolayk] this needs to be rewritten as follows
    //for each i : FunctionCall->get_inputs.at(i).get_tensor_view_type ==
    //flatten(function_parms).at(i)
    if (get_input_ops().size() != function_params.size())
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    for (size_t i = 0; i < get_input_ops().size(); i++)
    {
        if (nullptr == get_input_ops().at(i)->get_value_type())
        {
            throw ngraph_error("Function call argument is missing type.");
        }

        if (nullptr == function_params.at(i)->get_value_type())
        {
            throw ngraph_error("Function parameter is missing type.");
        }

        if (*(get_input_ops().at(i)->get_value_type()) !=
            *(function_params.at(i)->get_value_type()))
        {
            throw ngraph_error("Function argument type mismatch.");
        }
    }

    if (!std::dynamic_pointer_cast<XLAFunction>(m_function) &&
        m_function->get_results().size() >
            1) //TODO: we don't expect regular functions with multiple outputs just yet
    {
        throw "Regular functions with multiple outputs NYI!";
    }
    auto f_result_type = m_function->get_result_types().at(0);

    set_value_type_checked(f_result_type);
}
