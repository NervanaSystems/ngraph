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

using namespace std;
using namespace ngraph::op;

void FunctionCall::propagate_types()
{
    auto& function_params = m_function->get_parameters();

    if (m_arguments.size() != function_params.size())
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    for (size_t i = 0; i < m_arguments.size(); i++)
    {
        if (nullptr == m_arguments.at(i)->get_value_type())
        {
            throw ngraph_error("Function call argument is missing type.");
        }

        if (nullptr == function_params.at(i)->get_value_type())
        {
            throw ngraph_error("Function parameter is missing type.");
        }

        if (*(m_arguments.at(i)->get_value_type()) != *(function_params.at(i)->get_value_type()))
        {
            throw ngraph_error("Function argument type mismatch.");
        }
    }

    auto f_result_type = m_function->get_result_type();

    set_value_type_checked(f_result_type);
}
