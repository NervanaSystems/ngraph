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

#include "ngraph/op/max_pool.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

const string op::MaxPool::type_name{"MaxPool"};

op::MaxPool::MaxPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     const PadType& pad_type,
                     bool ceil_mode)
    : Op({arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_pad_type(pad_type)
    , m_ceil_mode(ceil_mode)
{
    constructor_validate_and_infer_types();
}

op::MaxPool::MaxPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     const PadType& pad_type)
    : MaxPool(
          arg, window_shape, window_movement_strides, padding_below, padding_above, pad_type, false)
{
}

op::MaxPool::MaxPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above)
    : MaxPool(arg,
              window_shape,
              window_movement_strides,
              padding_below,
              padding_above,
              PadType::EXPLICIT)
{
}

void op::MaxPool::validate_and_infer_types()
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
                                                  true,
                                                  m_ceil_mode));
}

op::MaxPool::MaxPool(const Output<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : MaxPool(arg, window_shape, window_movement_strides, Shape(), Shape())
{
}

op::MaxPool::MaxPool(const Output<Node>& arg, const Shape& window_shape)
    : MaxPool(arg, window_shape, Strides(), Shape(), Shape())
{
}

shared_ptr<Node> op::MaxPool::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MaxPool>(new_args.at(0),
                                m_window_shape,
                                m_window_movement_strides,
                                m_padding_below,
                                m_padding_above,
                                m_pad_type,
                                m_ceil_mode);
}

const string op::MaxPoolBackprop::type_name{"MaxPoolBackprop"};

op::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                     const Output<Node>& delta,
                                     const Shape& window_shape,
                                     const Strides& window_movement_strides,
                                     const Shape& padding_below,
                                     const Shape& padding_above)
    : Op({arg_forward, delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    constructor_validate_and_infer_types();
}

op::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                     const Output<Node>& delta,
                                     const Output<Node>& result_forward,
                                     const Shape& window_shape,
                                     const Strides& window_movement_strides,
                                     const Shape& padding_below,
                                     const Shape& padding_above)
    : Op({arg_forward, delta, result_forward})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    constructor_validate_and_infer_types();
}

void op::MaxPoolBackprop::validate_and_infer_types()
{
    element::Type forward_arg_et = get_input_element_type(0);
    element::Type delta_et = get_input_element_type(1);

    element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, forward_arg_et, delta_et),
                          "Element types for forward argument (",
                          forward_arg_et,
                          ") and delta (",
                          delta_et,
                          ") do not match.");

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    const PartialShape& forward_arg_shape = get_input_partial_shape(0);

    PartialShape forward_result_shape = infer_batched_pooling_forward(this,
                                                                      forward_arg_shape,
                                                                      padding_below,
                                                                      padding_above,
                                                                      m_window_shape,
                                                                      m_window_movement_strides,
                                                                      true);

    const PartialShape& delta_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    // TODO(amprocte): We may technically be able to infer some extra information from
    // forward_result_shape that was not present in the forward arg shape---namely batch size and
    // channel count. Merge that info in.
    set_output_type(0, get_input_element_type(0), forward_arg_shape);
}

shared_ptr<Node> op::MaxPoolBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (this->get_input_size() == 3)
    {
        return make_shared<op::MaxPoolBackprop>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_window_shape,
                                                m_window_movement_strides,
                                                m_padding_below,
                                                m_padding_above);
    }

    return make_shared<op::MaxPoolBackprop>(new_args.at(0),
                                            new_args.at(1),
                                            m_window_shape,
                                            m_window_movement_strides,
                                            m_padding_below,
                                            m_padding_above);
}

void op::MaxPool::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    if (m_ceil_mode)
    {
        throw ngraph_error("Autodiff not supported on MaxPool with ceil_mode set");
    }

    auto delta = deltas.at(0);

    auto operand = input(0).get_source_output();
    auto backprop =
        make_shared<op::MaxPoolBackprop>(operand,
                                         delta,
                                         static_pointer_cast<op::MaxPool>(shared_from_this()),
                                         m_window_shape,
                                         m_window_movement_strides,
                                         m_padding_below,
                                         m_padding_above);

    adjoints.add_delta(operand, backprop);
}
