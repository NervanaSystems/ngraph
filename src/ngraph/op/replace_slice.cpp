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
    : Op("ReplaceSlice", check_single_output_args({arg0, arg1}))
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    constructor_validate_and_infer_types();

    check_args();
}

op::ReplaceSlice::ReplaceSlice(const shared_ptr<Node>& arg0,
                               const shared_ptr<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds)
    : Op("ReplaceSlice", check_single_output_args({arg0, arg1}))
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides(lower_bounds.size(), 1))
{
    constructor_validate_and_infer_types();

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

    NODE_VALIDATION_ASSERT(this, input_0_shape.size() == input_1_shape.size())
        << "Argument ranks do not match (arg0 shape: " << input_0_shape
        << ", arg1 shape: " << input_1_shape << ").";

    NODE_VALIDATION_ASSERT(this, input_0_element_type == input_1_element_type)
        << "Argument element types do not match (arg0 element type: " << input_0_element_type
        << ", arg1 element type: " << input_1_element_type << ").";

    NODE_VALIDATION_ASSERT(this, m_lower_bounds.size() == input_0_shape.size())
        << "Rank of lower bounds (" << m_lower_bounds.size() << ") does not match rank "
        << "of argument (" << input_0_shape.size() << ") (lower bounds: " << m_lower_bounds
        << ", argument shape: " << input_0_shape << ").";

    NODE_VALIDATION_ASSERT(this, m_upper_bounds.size() == input_0_shape.size())
        << "Rank of upper bounds (" << m_upper_bounds.size() << ") does not match rank "
        << "of argument (" << input_0_shape.size() << ") (upper bounds: " << m_upper_bounds
        << ", argument shape: " << input_0_shape << ").";

    NODE_VALIDATION_ASSERT(this, m_strides.size() == input_0_shape.size())
        << "Rank of strides (" << m_strides.size() << ") does not match rank "
        << "of argument (" << input_0_shape.size() << ") (strides: " << m_strides
        << ", argument shape: " << input_0_shape << ").";

    Shape slice_shape;

    for (size_t i = 0; i < input_0_shape.size(); i++)
    {
        NODE_VALIDATION_ASSERT(this, m_upper_bounds[i] <= input_0_shape[i])
            << "Upper bound for slice at axis " << i << " is out of range "
            << "(upper bounds: " << m_upper_bounds << ", argument shape: " << input_0_shape << ").";

        NODE_VALIDATION_ASSERT(this, m_lower_bounds[i] <= m_upper_bounds[i])
            << "Lower bound for slice is greater than upper bound at axis " << i
            << " (lower bounds: " << m_lower_bounds << ", upper bounds: " << m_upper_bounds << ").";

        NODE_VALIDATION_ASSERT(this, m_strides[i] != 0) << "Stride for slice is zero at axis " << i
                                                        << " (strides: " << m_strides << ").";

        size_t slice_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        slice_axis_size =
            slice_axis_size / m_strides[i] + ((slice_axis_size % m_strides[i] == 0) ? 0 : 1);
        slice_shape.push_back(slice_axis_size);
    }

    NODE_VALIDATION_ASSERT(this, input_1_shape == slice_shape)
        << "Shape of replacement tensor (" << input_1_shape << ") does not match the slice shape "
        << "(" << slice_shape << ").";

    set_output_type(0, input_0_element_type, input_0_shape);
}

shared_ptr<Node> op::ReplaceSlice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
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
