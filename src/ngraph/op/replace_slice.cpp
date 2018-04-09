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

#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

op::ReplaceSlice::ReplaceSlice(const shared_ptr<Node>& arg0,
                               const shared_ptr<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Strides& strides)
    : RequiresTensorViewArgs("ReplaceSlice", {arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    check_args();
}

op::ReplaceSlice::ReplaceSlice(const shared_ptr<Node>& arg0,
                               const shared_ptr<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds)
    : RequiresTensorViewArgs("ReplaceSlice", {arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides(lower_bounds.size(), 1))
{
    check_args();
}

void op::ReplaceSlice::check_args()
{
    auto& input_0 = get_inputs().at(0);
    auto& input_0_shape = input_0.get_shape();
    auto& input_0_element_type = input_0.get_element_type();

    auto& input_1 = get_inputs().at(1);
    auto& input_1_shape = input_1.get_shape();
    auto& input_1_element_type = input_1.get_element_type();

    if (input_0_shape.size() != input_1_shape.size())
    {
        throw ngraph_error("Replace-slice argument ranks do not match");
    }

    if (input_0_element_type != input_1_element_type)
    {
        throw ngraph_error("Element types for replace-slice arguments do not match");
    }

    if (m_lower_bounds.size() != input_0_shape.size())
    {
        throw ngraph_error(
            "Number of lower bounds provided for slice does not match number of input axes");
    }

    if (m_upper_bounds.size() != input_0_shape.size())
    {
        throw ngraph_error(
            "Number of upper bounds provided for slice does not match number of input axes");
    }

    if (m_strides.size() != input_0_shape.size())
    {
        throw ngraph_error(
            "Number of strides provided for slice does not match number of input axes");
    }

    Shape slice_shape;

    for (size_t i = 0; i < input_0_shape.size(); i++)
    {
        if (m_upper_bounds[i] > input_0_shape[i])
        {
            throw ngraph_error("Upper bound for slice is out of range");
        }

        if (m_lower_bounds[i] > m_upper_bounds[i])
        {
            throw ngraph_error("Lower bound for slice is greater than upper bound");
        }

        if (0 == m_strides[i])
        {
            throw ngraph_error("Stride for slice is zero");
        }

        size_t slice_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        slice_axis_size =
            slice_axis_size / m_strides[i] + ((slice_axis_size % m_strides[i] == 0) ? 0 : 1);
        slice_shape.push_back(slice_axis_size);
    }

    if (input_1_shape != slice_shape)
    {
        throw ngraph_error("Shape of replacement tensor does not match slice shape");
    }

    set_value_type_checked(input_0_element_type, input_0_shape);
}

shared_ptr<Node> op::ReplaceSlice::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ReplaceSlice>(
        new_args.at(0), new_args.at(1), m_lower_bounds, m_upper_bounds, m_strides);
}

void op::ReplaceSlice::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_inputs().at(0).get_output().get_node();
    auto& y_input = get_inputs().at(1);
    auto y = y_input.get_output().get_node();
    auto& y_element_type = y_input.get_element_type();
    auto y_shape = y_input.get_shape();

    auto zeros_shaped_like_y = op::Constant::create(y_element_type, y_shape, {0.0});

    adjoints.add_delta(x,
                       make_shared<op::ReplaceSlice>(
                           delta, zeros_shaped_like_y, m_lower_bounds, m_upper_bounds, m_strides));
    adjoints.add_delta(y, make_shared<op::Slice>(delta, m_lower_bounds, m_upper_bounds, m_strides));
}
