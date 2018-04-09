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

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::AvgPool::AvgPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above,
                     bool include_padding_in_avg_computation)
    : RequiresTensorViewArgs("AvgPool", {arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
    auto& arg_shape = get_input_shape(0);

    //
    // Make sure arg: NCDi for some Di of rank>0, N != 0, C != 0.
    //
    if (arg_shape.size() < 3)
    {
        throw ngraph_error(
            "Average-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }

    size_t batch_size = arg_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Average-pool data batch size is zero.");
    }

    size_t channel_count = arg_shape[1];
    if (channel_count == 0)
    {
        throw ngraph_error("Average-pool requires at least one feature channel.");
    }

    size_t spatial_dimension_count = arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    if (window_shape.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool window shape rank does not match number of spatial dimensions.");
    }

    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool window movement stride rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_below.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool below-padding rank does not match number of spatial dimensions.");
    }

    if (padding_above.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool above-padding rank does not match number of spatial dimensions.");
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
            throw ngraph_error("Average-pool input spatial dimension is zero even after padding.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] == 0)
        {
            throw ngraph_error("Average-pool window shape has a zero-length axis.");
        }
    }

    //
    // Make sure the pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] > input_item_virtual_shape[i])
        {
            throw ngraph_error(
                "Average-pool window shape is larger than the spatial dimensions even after "
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
            throw ngraph_error("Average-pool window axis movement stride is zero.");
        }
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - window_shape[i] + 1,
                                             window_movement_strides[i]));
    }

    //
    // Make sure we're not going to have to compute average over an empty set of tensor elements.
    // That will happen if the sliding window ever resides entirely over the padding area AND
    // we're planning to disregard padding when computing the window's average.
    //
    if (!include_padding_in_avg_computation)
    {
        for (size_t i = 0; i < spatial_dimension_count; i++)
        {
            const size_t dim_virtual_size = input_item_virtual_shape[i];
            const size_t dim_window_size = window_shape[i];
            const size_t dim_stride = window_movement_strides[i];
            const size_t dim_padding_below = padding_below[i];
            const size_t dim_padding_above = padding_above[i];

            // Checking the lower edge of each dimension is easy, because there's no mystery
            // regarding the window's lower-edge placement...
            if ((dim_padding_below > 0) && (dim_window_size <= dim_padding_below))
            {
                throw ngraph_error(
                    "Average-pool window will sometimes reside entirely within the padding-below "
                    "region, but this average-pool op disregards padding elements.");
            }

            // Now check the upper-bound...
            {
                const size_t dim_num_strides = (dim_virtual_size - dim_window_size) / dim_stride;
                const size_t dim_window_max_lower_offset = dim_num_strides * dim_stride;
                const size_t dim_padding_above_start_offset = dim_virtual_size - dim_padding_above;

                if ((dim_padding_above > 0) &&
                    (dim_window_max_lower_offset >= dim_padding_above_start_offset))
                {
                    throw ngraph_error(
                        "Average-pool window will sometimes reside entirely within the "
                        "padding-above "
                        "region, but this average-pool op disregards padding elements.");
                }
            }
        }
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
        throw ngraph_error("Average-pool data batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Average-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }
    return Shape(arg_shape.size() - 2, 0);
}

op::AvgPool::AvgPool(const shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : AvgPool(arg,
              window_shape,
              window_movement_strides,
              default_padding(arg),
              default_padding(arg),
              false)
{
}

static Strides default_strides(const shared_ptr<Node>& arg)
{
    if (arg->get_outputs().size() != 1)
    {
        throw ngraph_error("Average-pool data batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Average-pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }
    return Strides(arg_shape.size() - 2, 1);
}

op::AvgPool::AvgPool(const shared_ptr<Node>& arg, const Shape& window_shape)
    : AvgPool(arg,
              window_shape,
              default_strides(arg),
              default_padding(arg),
              default_padding(arg),
              false)
{
}

shared_ptr<Node> op::AvgPool::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

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
    : RequiresTensorViewArgs("AvgPoolBackprop", {delta})
    , m_forward_arg_shape(forward_arg_shape)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
    // --
    // TODO: de-duplicate this code from AvgPool::AvgPool.
    // --

    auto& delta_shape = get_input_shape(0);

    //
    // Make sure arg: NCDi for some Di of rank>0, N != 0, C != 0.
    //
    if (forward_arg_shape.size() < 3)
    {
        throw ngraph_error(
            "Average-pool backprop: data batch shape must have rank of at least 3 (one batch axis, "
            "one channel axis, at least one spatial dimension).");
    }

    size_t batch_size = forward_arg_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Average-pool backprop: data batch size is zero.");
    }

    size_t channel_count = forward_arg_shape[1];
    if (channel_count == 0)
    {
        throw ngraph_error("Average-pool backprop: requires at least one feature channel.");
    }

    size_t spatial_dimension_count = forward_arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    if (window_shape.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool backprop: window shape rank does not match number of spatial "
            "dimensions.");
    }

    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool backprop: window movement stride rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_below.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool backprop: below-padding rank does not match number of spatial "
            "dimensions.");
    }

    if (padding_above.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Average-pool backprop: above-padding rank does not match number of spatial "
            "dimensions.");
    }

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = forward_arg_shape[1 + 1 + i];
        size_t virtual_dim_size = padding_below[i] + dim_size + padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);

        if (virtual_dim_size == 0)
        {
            throw ngraph_error(
                "Average-pool backprop: data batch spatial dimension is zero even after padding.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] == 0)
        {
            throw ngraph_error("Average-pool backprop: window shape has a zero-length axis.");
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
                "Average-pool backprop: window shape is larger than the spatial dimensions even "
                "after "
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
            throw ngraph_error("Average-pool backprop: window axis movement stride is zero.");
        }
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - window_shape[i] + 1,
                                             window_movement_strides[i]));
    }

    //
    // Make sure we're not going to have to compute average over an empty set of tensor elements.
    // That will happen if the sliding window ever resides entirely over the padding area AND
    // we're planning to disregard padding when computing the window's average.
    //
    if (!include_padding_in_avg_computation)
    {
        for (size_t i = 0; i < spatial_dimension_count; i++)
        {
            const size_t dim_virtual_size = input_item_virtual_shape[i];
            const size_t dim_window_size = window_shape[i];
            const size_t dim_stride = window_movement_strides[i];
            const size_t dim_padding_below = padding_below[i];
            const size_t dim_padding_above = padding_above[i];

            // Checking the lower edge of each dimension is easy, because there's no mystery
            // regarding the window's lower-edge placement...
            if ((dim_padding_below > 0) && (dim_window_size <= dim_padding_below))
            {
                throw ngraph_error(
                    "AvgPoolBackprop window will sometimes reside entirely within the "
                    "padding-below region, but the op disregards padding elements.");
            }

            // Now check the upper-bound...
            {
                const size_t dim_num_strides = (dim_virtual_size - dim_window_size) / dim_stride;
                const size_t dim_window_max_lower_offset = dim_num_strides * dim_stride;
                const size_t dim_padding_above_start_offset = dim_virtual_size - dim_padding_above;

                if ((dim_padding_above > 0) &&
                    (dim_window_max_lower_offset >= dim_padding_above_start_offset))
                {
                    throw ngraph_error(
                        "AvgPoolBackprop window will sometimes reside entirely within the "
                        "padding-above region, but the op disregards padding elements.");
                }
            }
        }
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
        throw ngraph_error(
            "Average-pool backprop: forward result shape does not match delta shape.");
    }

    set_value_type_checked(get_input_element_type(0), forward_arg_shape);
}

shared_ptr<Node> op::AvgPoolBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

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

    auto operand = get_input_op(0);
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
