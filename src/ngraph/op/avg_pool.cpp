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

    //
    // Make sure batch size and channel count are not zero, and that we have at least one spatial
    // dimension (in other words, that arg has shape NCDi for some Di of rank>0, N != 0, C != 0).
    //
    NODE_VALIDATION_ASSERT(this, arg_shape.size() >= 3)
        << "Data input shape does not have rank of at least 3 (data input shape: " << arg_shape
        << ").";

    size_t batch_size = arg_shape[0];
    NODE_VALIDATION_ASSERT(this, batch_size != 0)
        << "Data batch size is zero (data input shape: " << arg_shape << ").";

    size_t channel_count = arg_shape[1];
    NODE_VALIDATION_ASSERT(this, channel_count != 0)
        << "Channel count is zero (data input shape: " << arg_shape << ").";

    size_t spatial_dimension_count = arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    NODE_VALIDATION_ASSERT(this, m_window_shape.size() == spatial_dimension_count)
        << "Window shape rank does not match number of spatial dimensions (window shape: "
        << m_window_shape << ", data input shape: " << arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_window_movement_strides.size() == spatial_dimension_count)
        << "Window movement stride rank does not match number of spatial dimensions (window "
           "movement strides: "
        << m_window_movement_strides << ", data input shape: " << arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_padding_below.size() == spatial_dimension_count)
        << "Below-padding rank does not match number of spatial dimensions (padding below: "
        << m_padding_below << ", data input shape: " << arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_padding_above.size() == spatial_dimension_count)
        << "Above-padding rank does not match number of spatial dimensions (padding above: "
        << m_padding_above << ", data input shape: " << arg_shape << ").";

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = arg_shape[1 + 1 + i];
        size_t virtual_dim_size = m_padding_below[i] + dim_size + m_padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);
    }

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, input_item_virtual_shape[i] != 0)
            << "Data input spatial dimension " << i
            << " has zero length even after padding (virtual shape of input item: "
            << input_item_virtual_shape << ").";
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_shape[i] != 0)
            << "Window shape dimension " << i
            << " has zero length (window shape: " << m_window_shape << ").";
    }

    //
    // Make sure the pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_shape[i] <= input_item_virtual_shape[i])
            << "Window shape after padding is larger than the spatial dimensions (window shape: "
            << m_window_shape << ", virtual shape of input item: " << input_item_virtual_shape
            << ").";
    }

    //
    // Compute output item shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    Shape output_item_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_movement_strides[i] != 0)
            << "Window movement strides dimension " << i
            << " has zero length (window movement strides: " << m_window_movement_strides << ").";
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - m_window_shape[i] + 1,
                                             m_window_movement_strides[i]));
    }

    //
    // Make sure we're not going to have to compute average over an empty set of tensor elements.
    // That will happen if the sliding window ever resides entirely over the padding area AND
    // we're planning to disregard padding when computing the window's average.
    //
    if (!m_include_padding_in_avg_computation)
    {
        for (size_t i = 0; i < spatial_dimension_count; i++)
        {
            const size_t dim_virtual_size = input_item_virtual_shape[i];
            const size_t dim_window_size = m_window_shape[i];
            const size_t dim_stride = m_window_movement_strides[i];
            const size_t dim_padding_below = m_padding_below[i];
            const size_t dim_padding_above = m_padding_above[i];

            // Checking the lower edge of each dimension is easy, because there's no mystery
            // regarding the window's lower-edge placement...
            NODE_VALIDATION_ASSERT(this,
                                   dim_padding_below == 0 || dim_window_size > dim_padding_below)
                << "Window will sometimes reside entirely within the below-padding region, but"
                << " include_padding_in_avg_computation was not set (padding below: "
                << m_padding_below << ", window shape: " << m_window_shape << ").";

            // Now check the upper-bound...
            {
                const size_t dim_num_strides = (dim_virtual_size - dim_window_size) / dim_stride;
                const size_t dim_window_max_lower_offset = dim_num_strides * dim_stride;
                const size_t dim_padding_above_start_offset = dim_virtual_size - dim_padding_above;

                NODE_VALIDATION_ASSERT(this,
                                       dim_padding_above == 0 ||
                                           dim_window_max_lower_offset <
                                               dim_padding_above_start_offset)
                    << "Window will sometimes reside entirely within the above-padding region, but"
                    << " include_padding_in_avg_computation was not set (padding above: "
                    << m_padding_above << ", window shape: " << m_window_shape << ").";
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

    set_output_type(0, get_input_element_type(0), result_shape);
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
    // --
    // TODO: de-duplicate this code from AvgPool::AvgPool.
    // --

    auto& delta_shape = get_input_shape(0);

    //
    // Make sure batch size and channel count are not zero, and that we have at least one spatial
    // dimension (in other words, that arg has shape NCDi for some Di of rank>0, N != 0, C != 0).
    //
    NODE_VALIDATION_ASSERT(this, m_forward_arg_shape.size() >= 3)
        << "Forward input shape does not have rank of at least 3 (forward input shape: "
        << m_forward_arg_shape << ").";

    size_t batch_size = m_forward_arg_shape[0];
    NODE_VALIDATION_ASSERT(this, batch_size != 0)
        << "Data batch size is zero (forward input shape: " << m_forward_arg_shape << ").";

    size_t channel_count = m_forward_arg_shape[1];
    NODE_VALIDATION_ASSERT(this, channel_count != 0)
        << "Channel count is zero (forward input shape: " << m_forward_arg_shape << ").";

    size_t spatial_dimension_count = m_forward_arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and padding have same rank as Di.
    //
    NODE_VALIDATION_ASSERT(this, m_window_shape.size() == spatial_dimension_count)
        << "Window shape rank does not match number of spatial dimensions (window shape: "
        << m_window_shape << ", forward input shape: " << m_forward_arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_window_movement_strides.size() == spatial_dimension_count)
        << "Window movement stride rank does not match number of spatial dimensions (window "
           "movement strides: "
        << m_window_movement_strides << ", forward input shape: " << m_forward_arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_padding_below.size() == spatial_dimension_count)
        << "Below-padding rank does not match number of spatial dimensions (padding below: "
        << m_padding_below << ", forward input shape: " << m_forward_arg_shape << ").";
    NODE_VALIDATION_ASSERT(this, m_padding_above.size() == spatial_dimension_count)
        << "Above-padding rank does not match number of spatial dimensions (padding above: "
        << m_padding_above << ", forward input shape: " << m_forward_arg_shape << ").";

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = m_forward_arg_shape[1 + 1 + i];
        size_t virtual_dim_size = m_padding_below[i] + dim_size + m_padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);
    }

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, input_item_virtual_shape[i] != 0)
            << "Forward input spatial dimension " << i
            << " has zero length even after padding (virtual shape of input item: "
            << input_item_virtual_shape << ").";
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_shape[i] != 0)
            << "Window shape dimension " << i
            << " has zero length (window shape: " << m_window_shape << ").";
    }

    //
    // Make sure the pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_shape[i] <= input_item_virtual_shape[i])
            << "Window shape after padding is larger than the spatial dimensions (window shape: "
            << m_window_shape << ", virtual shape of input item: " << input_item_virtual_shape
            << ").";
    }

    //
    // Compute output item shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    Shape output_item_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(this, m_window_movement_strides[i] != 0)
            << "Window movement strides dimension " << i
            << " has zero length (window movement strides: " << m_window_movement_strides << ").";
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - m_window_shape[i] + 1,
                                             m_window_movement_strides[i]));
    }

    //
    // Make sure we're not going to have to compute average over an empty set of tensor elements.
    // That will happen if the sliding window ever resides entirely over the padding area AND
    // we're planning to disregard padding when computing the window's average.
    //
    if (!m_include_padding_in_avg_computation)
    {
        for (size_t i = 0; i < spatial_dimension_count; i++)
        {
            const size_t dim_virtual_size = input_item_virtual_shape[i];
            const size_t dim_window_size = m_window_shape[i];
            const size_t dim_stride = m_window_movement_strides[i];
            const size_t dim_padding_below = m_padding_below[i];
            const size_t dim_padding_above = m_padding_above[i];

            // Checking the lower edge of each dimension is easy, because there's no mystery
            // regarding the window's lower-edge placement...
            NODE_VALIDATION_ASSERT(this,
                                   dim_padding_below == 0 || dim_window_size > dim_padding_below)
                << "Window will sometimes reside entirely within the below-padding region, but"
                << " include_padding_in_avg_computation was not set (padding below: "
                << m_padding_below << ", window shape: " << m_window_shape << ").";

            // Now check the upper-bound...
            {
                const size_t dim_num_strides = (dim_virtual_size - dim_window_size) / dim_stride;
                const size_t dim_window_max_lower_offset = dim_num_strides * dim_stride;
                const size_t dim_padding_above_start_offset = dim_virtual_size - dim_padding_above;

                NODE_VALIDATION_ASSERT(this,
                                       dim_padding_above == 0 ||
                                           dim_window_max_lower_offset <
                                               dim_padding_above_start_offset)
                    << "Window will sometimes reside entirely within the above-padding region, but"
                    << " include_padding_in_avg_computation was not set (padding above: "
                    << m_padding_above << ", window shape: " << m_window_shape << ").";
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

    NODE_VALIDATION_ASSERT(this, forward_result_shape == delta_shape)
        << "Inferred forward output shape does not match delta shape (inferred forward output "
           "shape: "
        << forward_result_shape << ", delta shape: " << delta_shape << ").";

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
