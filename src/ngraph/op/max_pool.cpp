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

#include "ngraph/op/max_pool.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::MaxPool::MaxPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above)
    : RequiresTensorViewArgs("MaxPool", {arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    auto& arg_shape = get_input_shape(0);

    //
    // Make sure arg: NCDi for some Di of rank>0, N != 0, C != 0.
    //
    if (arg_shape.size() < 3)
    {
        throw ngraph_error(
            "Max-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }

    size_t batch_size = arg_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Max-pool data batch size is zero.");
    }

    size_t channel_count = arg_shape[1];
    if (channel_count == 0)
    {
        throw ngraph_error("Max-pool requires at least one feature channel.");
    }

    size_t spatial_dimension_count = arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    if (window_shape.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool window shape rank does not match number of spatial dimensions.");
    }

    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool window movement stride rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_below.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool below-padding rank does not match number of spatial dimensions.");
    }

    if (padding_above.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool above-padding rank does not match number of spatial dimensions.");
    }

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = arg_shape[1 + 1 + i];
        size_t virtual_dim_size = padding_below[i] + dim_size + padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);

        if (virtual_dim_size == 0)
        {
            throw ngraph_error("Max-pool input spatial dimension is zero even after padding.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] == 0)
        {
            throw ngraph_error("Max-pool window shape has a zero-length axis.");
        }
    }

    //
    // Make the max pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] > input_item_virtual_shape[i])
        {
            throw ngraph_error(
                "Max-pool window shape is larger than the spatial dimensions even after "
                "padding.");
        }
    }

    //
    // Compute output item shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    Shape output_item_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error("Max-pool window axis movement stride is zero.");
        }
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - window_shape[i] + 1,
                                             window_movement_strides[i]));
    }

    //
    // Construct result shape: NCDo.
    //
    Shape result_shape(1 + 1 + spatial_dimension_count);
    result_shape[0] = batch_size;
    result_shape[1] = channel_count;
    copy(output_item_shape.begin(), output_item_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(get_input_element_type(0), result_shape);
}

static Shape default_padding(const shared_ptr<Node>& arg)
{
    if (arg->get_outputs().size() != 1)
    {
        throw ngraph_error("Max-pool data batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Max-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }
    return Shape(arg_shape.size() - 2, 0);
}

op::MaxPool::MaxPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : MaxPool(
          arg, window_shape, window_movement_strides, default_padding(arg), default_padding(arg))
{
}

static Strides default_strides(const shared_ptr<Node>& arg)
{
    if (arg->get_outputs().size() != 1)
    {
        throw ngraph_error("Max-pool data batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Max-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }
    return Strides(arg_shape.size() - 2, 1);
}

op::MaxPool::MaxPool(const shared_ptr<Node>& arg, const Shape& window_shape)
    : MaxPool(arg, window_shape, default_strides(arg), default_padding(arg), default_padding(arg))
{
}

shared_ptr<Node> op::MaxPool::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<MaxPool>(new_args.at(0),
                                m_window_shape,
                                m_window_movement_strides,
                                m_padding_below,
                                m_padding_above);
}

op::MaxPoolBackprop::MaxPoolBackprop(const shared_ptr<Node>& arg_forward,
                                     const shared_ptr<Node>& delta,
                                     const Shape& window_shape,
                                     const Strides& window_movement_strides,
                                     const Shape& padding_below,
                                     const Shape& padding_above,
                                     const shared_ptr<op::MaxPool>& forward_op)
    : RequiresTensorViewArgs("MaxPoolBackprop", {arg_forward, delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_forward_op(forward_op)
{
    // --
    // TODO: de-duplicate this code from MaxPool::MaxPool.
    // --

    if (get_input_element_type(0) != get_input_element_type(1))
    {
        throw ngraph_error("Max-pool backprop: data batch and delta element types do not match.");
    }

    auto& arg_forward_shape = get_input_shape(0);
    auto& delta_shape = get_input_shape(1);

    //
    // Make sure arg: NCDi for some Di of rank>0, N != 0, C != 0.
    //
    if (arg_forward_shape.size() < 3)
    {
        throw ngraph_error(
            "Max-pool backprop: data batch shape must have rank of at least 3 (one batch axis, "
            "one channel axis, at least one spatial dimension).");
    }

    size_t batch_size = arg_forward_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Max-pool backprop: data batch size is zero.");
    }

    size_t channel_count = arg_forward_shape[1];
    if (channel_count == 0)
    {
        throw ngraph_error("Max-pool backprop: requires at least one feature channel.");
    }

    size_t spatial_dimension_count = arg_forward_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    if (window_shape.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool backprop: window shape rank does not match number of spatial "
            "dimensions.");
    }

    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool backprop: window movement stride rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_below.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool backprop: below-padding rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_above.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max-pool backprop: above-padding rank does not match number of spatial "
            "dimensions.");
    }

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = arg_forward_shape[1 + 1 + i];
        size_t virtual_dim_size = padding_below[i] + dim_size + padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);

        if (virtual_dim_size == 0)
        {
            throw ngraph_error(
                "Max-pool backprop: data batch spatial dimension is zero even after padding.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] == 0)
        {
            throw ngraph_error("Max-pool backprop: window shape has a zero-length axis.");
        }
    }

    //
    // Make the max pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] > input_item_virtual_shape[i])
        {
            throw ngraph_error(
                "Max-pool backprop: window shape is larger than the spatial dimensions even after "
                "padding.");
        }
    }

    //
    // Compute output item shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    Shape output_item_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error("Max-pool backprop: window axis movement stride is zero.");
        }
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - window_shape[i] + 1,
                                             window_movement_strides[i]));
    }

    //
    // Construct result shape: NCDo.
    //
    Shape forward_result_shape(1 + 1 + spatial_dimension_count);
    forward_result_shape[0] = batch_size;
    forward_result_shape[1] = channel_count;
    copy(output_item_shape.begin(), output_item_shape.end(), forward_result_shape.begin() + 2);

    if (forward_result_shape != delta_shape)
    {
        throw ngraph_error("Max-pool backprop: forward result shape does not match delta shape.");
    }

    set_value_type_checked(get_input_element_type(0), arg_forward_shape);
}

shared_ptr<op::MaxPool> op::MaxPoolBackprop::get_forward_op() const
{
    return m_forward_op.lock();
}

shared_ptr<Node> op::MaxPoolBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    MaxPoolBackprop* mpbp = new MaxPoolBackprop(new_args.at(0),
                                                new_args.at(1),
                                                m_window_shape,
                                                m_window_movement_strides,
                                                m_padding_below,
                                                m_padding_above);
    return shared_ptr<op::MaxPoolBackprop>(mpbp);
}

void op::MaxPool::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto operand = get_argument(0);
    auto backprop =
        make_shared<op::MaxPoolBackprop>(operand,
                                         delta,
                                         m_window_shape,
                                         m_window_movement_strides,
                                         m_padding_below,
                                         m_padding_above,
                                         static_pointer_cast<op::MaxPool>(shared_from_this()));

    adjoints.add_delta(operand, backprop);
}
