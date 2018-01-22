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
            "Average-pool image batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one image dimension).");
    }

    m_batch_size = arg_shape[0];
    if (m_batch_size == 0)
    {
        throw ngraph_error("Average-pool image batch size is zero.");
    }

    m_channel_count = arg_shape[1];
    if (m_channel_count == 0)
    {
        throw ngraph_error("Average-pool requires at least one image depth channel.");
    }

    m_image_dimension_count = arg_shape.size() - 2;

    //
    // Make sure window shape, window movement strides, and  have same rank as Di.
    //
    if (m_window_shape.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Average-pool window shape rank does not match number of image dimensions.");
    }

    if (m_window_movement_strides.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Average-pool window movement stride rank does not match number of image dimensions.");
    }

    if (m_padding_below.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Average-pool below-padding rank does not match number of image dimensions.");
    }

    if (m_padding_above.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Average-pool above-padding rank does not match number of image dimensions.");
    }

    //
    // Extract input image shape Di and make sure all dimensions are larger than 0.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        size_t dim_size = arg_shape[1 + 1 + i];
        m_input_image_physical_shape.push_back(dim_size);
        m_input_image_virtual_shape.push_back(padding_below[i] + dim_size + padding_above[i]);

        if (m_input_image_virtual_shape[i] == 0)
        {
            throw ngraph_error("Average-pool input image dimension is zero even after padding.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (m_window_shape[i] == 0)
        {
            throw ngraph_error("Average-pool window shape has a zero-length axis.");
        }
    }

    //
    // Make the max pooling window fits within the image dimensions.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (m_window_shape[i] > m_input_image_virtual_shape[i])
        {
            throw ngraph_error(
                "Average-pool window shape is larger than the image even after padding.");
        }
    }

    //
    // Compute image output shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (m_window_movement_strides[i] == 0)
        {
            throw ngraph_error("Average-pool window axis movement stride is zero.");
        }
        m_output_image_shape.push_back(ceil_div(
            m_input_image_virtual_shape[i] - m_window_shape[i] + 1, m_window_movement_strides[i]));
    }

    //
    // Construct result shape: NCDo.
    //
    Shape result_shape(1 + 1 + m_image_dimension_count);
    result_shape[0] = m_batch_size;
    result_shape[1] = m_channel_count;
    std::copy(m_output_image_shape.begin(), m_output_image_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(get_input_element_type(0), result_shape);
}

static Shape default_padding(const std::shared_ptr<Node>& arg)
{
    if (arg->get_outputs().size() != 1)
    {
        throw ngraph_error("Average-pool image batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Average-pool image batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one image dimension).");
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
        throw ngraph_error("Average-pool image batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Average-pool image batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one image dimension).");
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
        rc &= m_channel_count == rhs.m_channel_count;
        rc &= m_input_image_physical_shape == rhs.m_input_image_physical_shape;
        rc &= m_input_image_virtual_shape == rhs.m_input_image_virtual_shape;
        rc &= m_output_image_shape == rhs.m_output_image_shape;
        rc &= m_batch_size == rhs.m_batch_size;
        rc &= m_image_dimension_count == rhs.m_image_dimension_count;
    }
    else
    {
        rc = false;
    }
    return rc;
}

op::AvgPoolBprop::AvgPoolBprop(const std::shared_ptr<Node>& arg,
                               const std::shared_ptr<Node>& delta,
                               const Shape& window_shape,
                               const Strides& window_movement_strides,
                               const Shape& padding_below,
                               const Shape& padding_above)
    : RequiresTensorViewArgs("AvgPoolBprop", {arg, delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    set_value_type_checked(get_input_element_type(0), arg->get_shape());
}

bool op::AvgPoolBprop::is_functionally_identical(const Node& other) const
{
    bool rc = true;
    if (Node::is_functionally_identical(other))
    {
        const AvgPoolBprop& rhs = dynamic_cast<const AvgPoolBprop&>(other);
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

void op::AvgPool::generate_adjoints(autodiff::Adjoints& adjoints,
                                    const std::shared_ptr<Node>& delta)
{
    auto operand = get_input_op(0);
    AvgPoolBprop* avpn = new AvgPoolBprop(operand,
                                          delta,
                                          m_window_shape,
                                          m_window_movement_strides,
                                          m_padding_below,
                                          m_padding_above);
    auto bprop = std::shared_ptr<op::AvgPoolBprop>(avpn);
    adjoints.add_delta(operand, bprop);
}
