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

#include "ngraph/ops/slice.hpp"

using namespace std;
using namespace ngraph::op;

void Slice::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_type = m_arguments.at(0)->get_value_type();
    if (nullptr == arg_type)
    {
        throw ngraph_error("Argument to slice is missing type.");
    }
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    if (nullptr == arg_tensor_view_type)
    {
        throw ngraph_error("Argument to slice is not a tensor view");
    }
    auto& arg_shape = arg_tensor_view_type->get_shape();

    if (m_lower_bounds.size() != arg_shape.size())
    {
        throw ngraph_error(
            "Number of lower bounds provided for slice does not match number of input axes");
    }

    if (m_upper_bounds.size() != arg_shape.size())
    {
        throw ngraph_error(
            "Number of upper bounds provided for slice does not match number of input axes");
    }

    if (m_step.size() != arg_shape.size())
    {
        throw ngraph_error(
            "Number of step axes provided for slice does not match number of input axes");
    }

    Shape result_shape;

    for (size_t i = 0; i < arg_shape.size(); i++)
    {
        if (m_upper_bounds[i] > arg_shape[i])
        {
            throw ngraph_error("Upper bound for slice is out of range");
        }

        if (m_lower_bounds[i] > m_upper_bounds[i])
        {
            throw ngraph_error("Lower bound for slice is greater than upper bound");
        }

        if (0 == m_step[i])
        {
            throw ngraph_error("Step distance for slice is zero");
        }

        size_t result_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        result_axis_size =
            result_axis_size / m_step[i] + ((result_axis_size % m_step[i] == 0) ? 0 : 1);
        result_shape.push_back(result_axis_size);
    }

    set_value_type_checked(
        make_shared<TensorViewType>(arg_tensor_view_type->get_element_type(), result_shape));
}
