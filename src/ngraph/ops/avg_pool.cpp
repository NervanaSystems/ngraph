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

#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::AvgPool::AvgPool(const std::shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above)
    : RequiresTensorViewArgs("AvgPool", {arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    auto& arg_shape = get_inputs().at(0).get_shape();

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
    // Make sure window shape, window movement strides, and  have same rank as Di.
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
    // Make the max pooling window fits within the spatial dimensions.
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
    // Construct result shape: NCDo.
    //
    Shape result_shape(1 + 1 + spatial_dimension_count);
    result_shape[0] = batch_size;
    result_shape[1] = channel_count;
    std::copy(output_item_shape.begin(), output_item_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(get_input_element_type(0), result_shape);
}

static Shape default_padding(const std::shared_ptr<Node>& arg)
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

op::AvgPool::AvgPool(const std::shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : AvgPool(
          arg, window_shape, window_movement_strides, default_padding(arg), default_padding(arg))
{
}

static Strides default_strides(const std::shared_ptr<Node>& arg)
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

op::AvgPool::AvgPool(const std::shared_ptr<Node>& arg, const Shape& window_shape)
    : AvgPool(arg, window_shape, default_strides(arg), default_padding(arg), default_padding(arg))
{
}

bool op::AvgPool::is_functionally_identical(const Node& other) const
{
    bool rc = true;
    if (Node::is_functionally_identical(other))
    {
        const AvgPool& rhs = dynamic_cast<const AvgPool&>(other);
        rc &= m_window_shape == rhs.m_window_shape;
        rc &= m_window_movement_strides == rhs.m_window_movement_strides;
        rc &= m_padding_below == rhs.m_padding_below;
        rc &= m_padding_above == rhs.m_padding_above;
    }
    else
    {
        rc = false;
    }
    return rc;
}

/*
void op::AvgPool::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
}
*/
