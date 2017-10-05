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

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/topological_sort.hpp"

using namespace std;
using namespace ngraph::op;

void Reduce::propagate_types()
{
    if (m_arguments.size() != 2)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_reductee_type = m_arguments.at(0)->get_value_type();
    if (nullptr == arg_reductee_type)
    {
        throw ngraph_error("Argument to reduce is missing type.");
    }
    auto arg_reductee_tensor_view_type =
        dynamic_pointer_cast<const TensorViewType>(arg_reductee_type);
    if (nullptr == arg_reductee_tensor_view_type)
    {
        throw ngraph_error("Argument to reduce is not a tensor view");
    }

    auto arg_init_type = m_arguments.at(1)->get_value_type();
    if (nullptr == arg_init_type)
    {
        throw ngraph_error("Argument for initial value is missing type.");
    }
    auto arg_init_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_init_type);
    if (nullptr == arg_init_tensor_view_type)
    {
        throw ngraph_error("Argument for initial value is not a tensor view");
    }
    if (arg_init_tensor_view_type->get_shape().size() != 0)
    {
        throw ngraph_error("Argument for initial value is not a scalar");
    }

    if (arg_init_tensor_view_type->get_element_type() !=
        arg_reductee_tensor_view_type->get_element_type())
    {
        throw ngraph_error("Element types for reductee and initial values do not match");
    }

    auto arg_reductee_shape = arg_reductee_tensor_view_type->get_shape();

    for (auto axis : m_reduction_axes)
    {
        if (axis >= arg_reductee_shape.size())
        {
            throw ngraph_error("Reduction axis is out of bounds");
        }
    }

    Shape result_shape;

    for (size_t i = 0; i < arg_reductee_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(arg_reductee_shape.at(i));
        }
    }

    auto f_params = m_reduction_function->get_parameters();

    if (f_params.size() != 2)
    {
        throw ngraph_error("Reduction function has wrong number of parameters (should be two)");
    }

    if (*(f_params.at(0)->get_value_type()) != *(arg_init_type))
    {
        throw ngraph_error("Argument 0 of reduction function has wrong type");
    }
    if (*(f_params.at(1)->get_value_type()) != *(arg_init_type))
    {
        throw ngraph_error("Argument 1 of reduction function has wrong type");
    }

    auto f_result_type = m_reduction_function->get_result_type();

    if (*(f_result_type) != *(arg_init_type))
    {
        throw ngraph_error("Return type from reduction function does not match expected");
    }

    set_value_type_checked(make_shared<TensorViewType>(
        arg_reductee_tensor_view_type->get_element_type(), result_shape));
}
