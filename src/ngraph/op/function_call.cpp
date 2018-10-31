//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/function_call.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

op::FunctionCall::FunctionCall(shared_ptr<Function> function, const NodeVector& args)
    : Node("FunctionCall", args)
    , m_function(function)
{
    set_output_size(m_function->get_output_size());
    constructor_validate_and_infer_types();
}

void op::FunctionCall::validate_and_infer_types()
{
    m_function->validate_nodes_and_infer_types();

    auto& function_params = m_function->get_parameters();

    NODE_VALIDATION_ASSERT(this, get_input_size() == function_params.size())
        << "Number of arguments (" << get_input_size() << ") does not match "
        << "number of function parameters (" << function_params.size() << ").";

    for (size_t i = 0; i < get_input_size(); i++)
    {
        NODE_VALIDATION_ASSERT(
            this,
            m_function->get_parameters()[i]->get_element_type().relaxes(get_input_element_type(i)))
            << "Element type mismatch for argument " << i << " (argument has type "
            << get_input_element_type(i) << ", function expects type "
            << m_function->get_parameters()[i]->get_element_type();

        NODE_VALIDATION_ASSERT(this,
                               m_function->get_parameters()[i]->get_output_partial_shape(0).relaxes(
                                   get_input_partial_shape(i)))
            << "Shape mismatch for argument " << i << " (argument has shape "
            << get_input_partial_shape(i) << ", function expects shape "
            << m_function->get_parameters()[i]->get_output_partial_shape(0);
    }

    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        set_output_type(
            i, m_function->get_output_element_type(i), m_function->get_output_partial_shape(i));
    }
}

shared_ptr<Node> op::FunctionCall::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    shared_ptr<FunctionCall> fc = make_shared<FunctionCall>(m_function, new_args);
    fc->m_function = clone_function(*m_function);
    return fc;
}

/// \return A singleton vector containing the function to be called.
vector<shared_ptr<Function>> op::FunctionCall::get_functions() const
{
    return vector<shared_ptr<Function>>{m_function};
}
