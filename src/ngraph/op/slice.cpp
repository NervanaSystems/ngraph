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

#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

op::Slice::Slice(const shared_ptr<Node>& arg,
                 const Coordinate& lower_bounds,
                 const Coordinate& upper_bounds,
                 const Strides& strides)
    : Op("Slice", check_single_output_args({arg}))
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    constructor_validate_and_infer_types();
}

op::Slice::Slice(const shared_ptr<Node>& arg,
                 const Coordinate& lower_bounds,
                 const Coordinate& upper_bounds)
    : Op("Slice", check_single_output_args({arg}))
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides())
{
    constructor_validate_and_infer_types();
}

void op::Slice::validate_and_infer_types()
{
    if (0 == m_strides.size())
    {
        m_strides = Strides(m_lower_bounds.size(), 1);
    }
    auto& input = get_inputs().at(0);
    auto& input_shape = input.get_shape();

    NODE_VALIDATION_ASSERT(this, m_lower_bounds.size() == input_shape.size())
        << "Rank of lower bounds (" << m_lower_bounds.size() << ") does not match rank "
        << "of argument (" << input_shape.size() << ") (lower bounds: " << m_lower_bounds
        << ", argument shape: " << input_shape << ").";

    NODE_VALIDATION_ASSERT(this, m_upper_bounds.size() == input_shape.size())
        << "Rank of upper bounds (" << m_upper_bounds.size() << ") does not match rank "
        << "of argument (" << input_shape.size() << ") (upper bounds: " << m_upper_bounds
        << ", argument shape: " << input_shape << ").";

    NODE_VALIDATION_ASSERT(this, m_strides.size() == input_shape.size())
        << "Rank of strides (" << m_strides.size() << ") does not match rank "
        << "of argument (" << input_shape.size() << ") (strides: " << m_strides
        << ", argument shape: " << input_shape << ").";

    Shape result_shape;

    for (size_t i = 0; i < input_shape.size(); i++)
    {
        NODE_VALIDATION_ASSERT(this, m_upper_bounds[i] <= input_shape[i])
            << "Upper bound for slice at axis " << i << " is out of range "
            << "(upper bounds: " << m_upper_bounds << ", argument shape: " << input_shape << ").";

        NODE_VALIDATION_ASSERT(this, m_lower_bounds[i] <= m_upper_bounds[i])
            << "Lower bound for slice is greater than upper bound at axis " << i
            << " (lower bounds: " << m_lower_bounds << ", upper bounds: " << m_upper_bounds << ").";

        NODE_VALIDATION_ASSERT(this, m_strides[i] != 0) << "Stride for slice is zero at axis " << i
                                                        << " (strides: " << m_strides << ").";

        size_t result_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        result_axis_size =
            result_axis_size / m_strides[i] + ((result_axis_size % m_strides[i] == 0) ? 0 : 1);
        result_shape.push_back(result_axis_size);
    }

    set_output_type(0, input.get_element_type(), result_shape);

    // Static value propagation.
    // Only two cases are handled: scalar (which is the identity) and vector.
    if (get_inputs()[0].get_output().has_static_value() && input_shape.size() == 0)
    {
        set_output_static_value(0, get_inputs()[0].get_output().get_static_value());
    }
    else if (get_inputs()[0].get_output().has_static_value() && input_shape.size() == 1)
    {
        auto& sv = get_inputs()[0].get_output().get_static_value();

        StaticValue sv_out;

        for (size_t i = m_lower_bounds[0]; i < m_upper_bounds[0]; i += m_strides[0])
        {
            sv_out.push_back(sv[i]);
        }

        set_output_static_value(0, sv_out);
    }
    else
    {
        clear_output_static_value(0);
    }
}

shared_ptr<Node> op::Slice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Slice>(new_args.at(0), m_lower_bounds, m_upper_bounds, m_strides);
}

void op::Slice::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_inputs().at(0).get_output().get_node();

    adjoints.add_delta_to_slice(x, delta, m_lower_bounds, m_upper_bounds, m_strides);
}
