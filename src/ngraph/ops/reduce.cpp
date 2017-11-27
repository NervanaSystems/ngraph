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

#include "ngraph/ops/reduce.hpp"
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
    auto arg_reductee_tensor_view_type = get_inputs().at(0).get_tensor_view_type();

    auto arg_init_tensor_view_type = get_inputs().at(1).get_tensor_view_type();
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

    if (*(f_params.at(0)->get_value_type()) != *(arg_init->get_value_type()))
    {
        throw ngraph_error("Argument 0 of reduction function has wrong type");
    }
    if (*(f_params.at(1)->get_value_type()) != *(arg_init->get_value_type()))
    {
        throw ngraph_error("Argument 1 of reduction function has wrong type");
    }

    auto f_result_type = m_reduction_function->get_result_types().at(0);

    if (*(f_result_type) != *(arg_init->get_value_type()))
    {
        throw ngraph_error("Return type from reduction function does not match expected");
    }

    set_value_type_checked(make_shared<TensorViewType>(
        arg_reductee_tensor_view_type->get_element_type(), result_shape));
}
