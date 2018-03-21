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

#include "ngraph/op/reduce.hpp"
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph;

op::Reduce::Reduce(const std::shared_ptr<Node>& arg_reductee,
                   const std::shared_ptr<Node>& arg_init,
                   const std::shared_ptr<Function>& reduction_function,
                   const AxisSet& reduction_axes)
    : RequiresTensorViewArgs("Reduce", {arg_reductee, arg_init})
    , m_reduction_function(reduction_function)
    , m_reduction_axes(reduction_axes)
{
    auto& input_reductee = get_inputs().at(0);

    auto& input_init = get_inputs().at(1);
    if (input_init.get_shape().size() != 0)
    {
        throw ngraph_error("Argument for initial value is not a scalar");
    }

    if (input_init.get_element_type() != input_reductee.get_element_type())
    {
        throw ngraph_error("Element types for reductee and initial values do not match");
    }

    auto input_reductee_shape = input_reductee.get_shape();

    for (auto axis : m_reduction_axes)
    {
        if (axis >= input_reductee_shape.size())
        {
            throw ngraph_error("Reduction axis is out of bounds");
        }
    }

    Shape result_shape;

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(input_reductee_shape.at(i));
        }
    }

    auto f_params = m_reduction_function->get_parameters();

    if (f_params.size() != 2)
    {
        throw ngraph_error("Reduction function has wrong number of parameters (should be two)");
    }

    if (!f_params.at(0)->has_same_type(arg_init))
    {
        throw ngraph_error("Argument 0 of reduction function has wrong type");
    }
    if (!f_params.at(1)->has_same_type(arg_init))
    {
        throw ngraph_error("Argument 1 of reduction function has wrong type");
    }

    if (m_reduction_function->get_output_size() > 1)
    {
        throw ngraph_error("Single-output reduce function was expected!");
    }
    if (m_reduction_function->get_output_element_type(0) != arg_init->get_element_type())
    {
        throw ngraph_error("Return element type from reduction function does not match expected");
    }
    if (m_reduction_function->get_output_shape(0) != Shape{})
    {
        throw ngraph_error("Return shape from reduction function is not a scalar");
    }

    add_output(input_reductee.get_element_type(), result_shape);
}
