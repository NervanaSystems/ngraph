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

#include "ngraph/ops/replace_slice.hpp"

using namespace std;
using namespace ngraph;

op::ReplaceSlice::ReplaceSlice(const std::shared_ptr<Node>& arg0,
                               const std::shared_ptr<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Shape& step)
    : RequiresTensorViewArgs("ReplaceSlice", {arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_step(step)
{
    check_args();
}

op::ReplaceSlice::ReplaceSlice(const std::shared_ptr<Node>& arg0,
                               const std::shared_ptr<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds)
    : RequiresTensorViewArgs("ReplaceSlice", {arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_step(Shape(lower_bounds.size(), 1))
{
    check_args();
}

void op::ReplaceSlice::check_args()
{
    auto arg0_tensor_view_type = get_inputs().at(0).get_tensor_view_type();
    auto& arg0_shape = arg0_tensor_view_type->get_shape();
    auto& arg0_element_type = arg0_tensor_view_type->get_element_type();

    auto arg1_tensor_view_type = get_inputs().at(1).get_tensor_view_type();
    auto& arg1_shape = arg1_tensor_view_type->get_shape();
    auto& arg1_element_type = arg1_tensor_view_type->get_element_type();

    if (arg0_shape.size() != arg1_shape.size())
    {
        throw ngraph_error("Replace-slice argument ranks do not match");
    }

    if (arg0_element_type != arg1_element_type)
    {
        throw ngraph_error("Element types for replace-slice arguments do not match");
    }

    if (m_lower_bounds.size() != arg0_shape.size())
    {
        throw ngraph_error(
            "Number of lower bounds provided for slice does not match number of input axes");
    }

    if (m_upper_bounds.size() != arg0_shape.size())
    {
        throw ngraph_error(
            "Number of upper bounds provided for slice does not match number of input axes");
    }

    if (m_step.size() != arg0_shape.size())
    {
        throw ngraph_error(
            "Number of step axes provided for slice does not match number of input axes");
    }

    Shape slice_shape;

    for (size_t i = 0; i < arg0_shape.size(); i++)
    {
        if (m_upper_bounds[i] > arg0_shape[i])
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

        size_t slice_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        slice_axis_size =
            slice_axis_size / m_step[i] + ((slice_axis_size % m_step[i] == 0) ? 0 : 1);
        slice_shape.push_back(slice_axis_size);
    }

    if (arg1_shape != slice_shape)
    {
        throw ngraph_error("Shape of replacement tensor does not match slice shape");
    }

    set_value_type_checked(arg0_tensor_view_type);
}
