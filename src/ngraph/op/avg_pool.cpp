//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

const string op::AvgPool::type_name{"AvgPool"};

op::AvgPool::AvgPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     bool include_padding_in_avg_computation,
                     const PadType& pad_type,
                     bool ceil_mode)
    : Op({arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
    , m_pad_type(pad_type)
    , m_ceil_mode(ceil_mode)
{
    constructor_validate_and_infer_types();
}

op::AvgPool::AvgPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     bool include_padding_in_avg_computation,
                     const PadType& pad_type)
    : AvgPool(arg,
              window_shape,
              window_movement_strides,
              padding_below,
              padding_above,
              include_padding_in_avg_computation,
              pad_type,
              false)
{
}

op::AvgPool::AvgPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     bool include_padding_in_avg_computation)
    : AvgPool(arg,
              window_shape,
              window_movement_strides,
              padding_below,
              padding_above,
              include_padding_in_avg_computation,
              PadType::EXPLICIT)
{
}

void op::AvgPool::validate_and_infer_types()
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = Shape(m_window_shape.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);

    if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
    {
        if (arg_shape.is_static())
        {
            CoordinateDiff padding_above, padding_below;
            infer_auto_padding(arg_shape.to_shape(),
                               m_window_shape,
                               m_window_movement_strides,
                               Strides(m_window_shape.size(), 1), // No dilation
                               m_pad_type,
                               padding_above,
                               padding_below);
            m_padding_above = Shape(padding_above.begin(), padding_above.end());
            m_padding_below = Shape(padding_below.begin(), padding_below.end());
        }
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
                                                  m_include_padding_in_avg_computation,
                                                  m_ceil_mode));
}

op::AvgPool::AvgPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : AvgPool(arg, window_shape, window_movement_strides, Shape(), Shape(), false)
{
}

op::AvgPool::AvgPool(const Output<Node>& arg, const Shape& window_shape)
    : AvgPool(arg, window_shape, Strides(), Shape(), Shape(), false)
{
}

const Shape& op::AvgPool::get_window_shape() const
{
    return m_window_shape;
}

void op::AvgPool::set_window_shape(const Shape& window_shape)
{
    m_window_shape = window_shape;
}

const Strides& op::AvgPool::get_window_movement_strides() const
{
    return m_window_movement_strides;
}

void op::AvgPool::set_window_movement_strides(const Strides& window_movement_strides)
{
    m_window_movement_strides = window_movement_strides;
}

const Shape& op::AvgPool::get_padding_below() const
{
    return m_padding_below;
}

void op::AvgPool::set_padding_below(const Shape& padding_below)
{
    m_padding_below = padding_below;
}

const Shape& op::AvgPool::get_padding_above() const
{
    return m_padding_above;
}

void op::AvgPool::set_padding_above(const Shape& padding_above)
{
    m_padding_above = padding_above;
}

bool op::AvgPool::get_include_padding_in_avg_computation() const
{
    return m_include_padding_in_avg_computation;
}

void op::AvgPool::set_include_padding_in_avg_computation(bool include_padding_in_avg_computation)
{
    m_include_padding_in_avg_computation = include_padding_in_avg_computation;
}

const op::PadType& op::AvgPool::get_pad_type() const
{
    return m_pad_type;
}

void op::AvgPool::set_pad_type(const op::PadType& pad_type)
{
    m_pad_type = pad_type;
}

bool op::AvgPool::get_ceil_mode() const
{
    return m_ceil_mode;
}

void op::AvgPool::set_ceil_mode(bool ceil_mode)
{
    m_ceil_mode = ceil_mode;
}

shared_ptr<Node> op::AvgPool::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<AvgPool>(new_args.at(0),
                                m_window_shape,
                                m_window_movement_strides,
                                m_padding_below,
                                m_padding_above,
                                m_include_padding_in_avg_computation,
                                m_pad_type,
                                m_ceil_mode);
}

shared_ptr<Node> op::AvgPool::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

const string op::AvgPoolBackprop::type_name("AvgPoolBackprop");

op::AvgPoolBackprop::AvgPoolBackprop(const Shape& forward_arg_shape,
                                     const shared_ptr<Node>& delta,
                                     const Shape& window_shape,
                                     const Strides& window_movement_strides,
                                     const Shape& padding_below,
                                     const Shape& padding_above,
                                     bool include_padding_in_avg_computation)
    : Op(check_single_output_args({delta}))
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
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      m_forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation);

    const PartialShape& delta_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    // TODO(amprocte): Once m_forward_arg_shape is allowed to be dynamic, we may technically be
    // able to infer some extra information from forward_result_shape that was not present in the
    // forward arg shape---namely batch size and channel count. Merge that info in.
    set_output_type(0, get_input_element_type(0), m_forward_arg_shape);
}

const Shape& op::AvgPoolBackprop::get_forward_arg_shape() const
{
    return m_forward_arg_shape;
}

void op::AvgPoolBackprop::set_forward_arg_shape(const Shape& forward_arg_shape)
{
    m_forward_arg_shape = forward_arg_shape;
}

const Shape& op::AvgPoolBackprop::get_window_shape() const
{
    return m_window_shape;
}

void op::AvgPoolBackprop::set_window_shape(const Shape& window_shape)
{
    m_window_shape = window_shape;
}

const Strides& op::AvgPoolBackprop::get_window_movement_strides() const
{
    return m_window_movement_strides;
}

void op::AvgPoolBackprop::set_window_movement_strides(const Strides& window_movement_strides)
{
    m_window_movement_strides = window_movement_strides;
}

const Shape& op::AvgPoolBackprop::get_padding_below() const
{
    return m_padding_below;
}

void op::AvgPoolBackprop::set_padding_below(const Shape& padding_below)
{
    m_padding_below = padding_below;
}

const Shape& op::AvgPoolBackprop::get_padding_above() const
{
    return m_padding_above;
}

void op::AvgPoolBackprop::set_padding_above(const Shape& padding_above)
{
    m_padding_above = padding_above;
}

bool op::AvgPoolBackprop::get_include_padding_in_avg_computation() const
{
    return m_include_padding_in_avg_computation;
}

void op::AvgPoolBackprop::set_include_padding_in_avg_computation(
    bool include_padding_in_avg_computation)
{
    m_include_padding_in_avg_computation = include_padding_in_avg_computation;
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
    if (m_ceil_mode)
    {
        throw ngraph_error("Autodiff not supported on AvgPool with ceil_mode set");
    }

    auto delta = deltas.at(0);

    auto operand = input_value(0);
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
