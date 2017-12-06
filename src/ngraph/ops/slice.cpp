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
using namespace ngraph;

op::Slice::Slice(const std::shared_ptr<Node>& arg,
                 const Coordinate& lower_bounds,
                 const Coordinate& upper_bounds,
                 const Strides& strides)
    : RequiresTensorViewArgs("Slice", {arg})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    check_args();
}

op::Slice::Slice(const std::shared_ptr<Node>& arg,
                 const Coordinate& lower_bounds,
                 const Coordinate& upper_bounds)
    : RequiresTensorViewArgs("Slice", {arg})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides(lower_bounds.size(), 1))
{
    check_args();
}

void op::Slice::check_args()
{
    auto arg_tensor_view_type = get_inputs().at(0).get_tensor_view_type();
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

    if (m_strides.size() != arg_shape.size())
    {
        throw ngraph_error(
            "Number of strides provided for slice does not match number of input axes");
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

        if (0 == m_strides[i])
        {
            throw ngraph_error("Strides distance for slice is zero");
        }

        size_t result_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        result_axis_size =
            result_axis_size / m_strides[i] + ((result_axis_size % m_strides[i] == 0) ? 0 : 1);
        result_shape.push_back(result_axis_size);
    }

    set_value_type_checked(
        make_shared<TensorViewType>(arg_tensor_view_type->get_element_type(), result_shape));
}

void op::Slice::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto x = get_inputs().at(0).get_output().get_node();

    adjoints.add_delta_to_slice(x, delta, m_lower_bounds, m_upper_bounds, m_strides);
}
