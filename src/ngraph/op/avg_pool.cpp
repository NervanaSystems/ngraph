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

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::AvgPool::AvgPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     bool include_padding_in_avg_computation)
    : Op("AvgPool", check_single_output_args({arg}))
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
    constructor_validate_and_infer_types();
}

void op::AvgPool::validate_and_infer_types()
{
    auto& arg_shape = get_input_shape(0);

    NODE_VALIDATION_ASSERT(this, arg_shape.size() >= 3)
        << "Data input shape does not have rank of at least 3 (data input shape: " << arg_shape
        << ").";

    if (0 == m_window_movement_strides.size() && arg_shape.size() > 2)
    {
        m_window_movement_strides = Strides(arg_shape.size() - 2, 1);
    }

    if (0 == m_padding_below.size() && arg_shape.size() > 2)
    {
        m_padding_below = Shape(arg_shape.size() - 2, 0);
    }

    if (0 == m_padding_above.size() && arg_shape.size() > 2)
    {
        m_padding_above = Shape(arg_shape.size() - 2, 0);
    }

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    set_output_type(0,
                    get_input_element_type(0),
                    infer_batched_pooling_forward(this,
                                                  arg_shape,
                                                  padding_below,
                                                  padding_above,
                                                  m_window_shape,
                                                  m_window_movement_strides,
                                                  m_include_padding_in_avg_computation));
}

op::AvgPool::AvgPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : AvgPool(arg, window_shape, window_movement_strides, Shape(), Shape(), false)
{
}

op::AvgPool::AvgPool(const shared_ptr<Node>& arg, const Shape& window_shape)
    : AvgPool(arg, window_shape, Strides(), Shape(), Shape(), false)
{
}

shared_ptr<Node> op::AvgPool::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<AvgPool>(new_args.at(0),
                                m_window_shape,
                                m_window_movement_strides,
                                m_padding_below,
                                m_padding_above,
                                m_include_padding_in_avg_computation);
}

op::AvgPoolBackprop::AvgPoolBackprop(const Shape& forward_arg_shape,
                                     const shared_ptr<Node>& delta,
                                     const Shape& window_shape,
                                     const Strides& window_movement_strides,
                                     const Shape& padding_below,
                                     const Shape& padding_above,
                                     bool include_padding_in_avg_computation)
    : Op("AvgPoolBackprop", check_single_output_args({delta}))
    , m_forward_arg_shape(forward_arg_shape)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
    constructor_validate_and_infer_types();
}

void op::AvgPoolBackprop::validate_and_infer_types()
{
    auto& delta_shape = get_input_shape(0);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    Shape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      m_forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation);

    NODE_VALIDATION_ASSERT(this, forward_result_shape == delta_shape)
        << "Inferred forward output shape does not match delta shape (inferred forward output "
        << "shape: " << forward_result_shape << ", delta shape: " << delta_shape << ").";

    set_output_type(0, get_input_element_type(0), m_forward_arg_shape);
}

shared_ptr<Node> op::AvgPoolBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    AvgPoolBackprop* avpn = new AvgPoolBackprop(m_forward_arg_shape,
                                                new_args.at(0),
                                                m_window_shape,
                                                m_window_movement_strides,
                                                m_padding_below,
                                                m_padding_above,
                                                m_include_padding_in_avg_computation);
    return shared_ptr<op::AvgPoolBackprop>(avpn);
}

void op::AvgPool::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto operand = get_argument(0);
    auto& operand_shape = get_input_shape(0);
    auto backprop = make_shared<op::AvgPoolBackprop>(operand_shape,
                                                     delta,
                                                     m_window_shape,
                                                     m_window_movement_strides,
                                                     m_padding_below,
                                                     m_padding_above,
                                                     m_include_padding_in_avg_computation);
    adjoints.add_delta(operand, backprop);
}
