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

#include "ngraph/ops/convolution.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const Shape& padding_below,
                             const Shape& padding_above,
                             const Strides& image_dilation_strides)
    : RequiresTensorViewArgs("Convolution", {image_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_image_dilation_strides(image_dilation_strides)
{
    auto& image_batch_shape = get_inputs().at(0).get_shape();
    auto& filters_shape = get_inputs().at(1).get_shape();

    //
    // Make sure image_batch: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    if (image_batch_shape.size() < 3)
    {
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }

    m_batch_size = image_batch_shape[0];
    if (m_batch_size == 0)
    {
        throw ngraph_error("Convolution image batch size is zero.");
    }

    m_input_channel_count = image_batch_shape[1];
    if (m_input_channel_count == 0)
    {
        throw ngraph_error("Convolution requires at least one input channel.");
    }

    m_image_dimension_count = image_batch_shape.size() - 2;

    //
    // Make sure filters: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    if (filters_shape.size() != 2 + m_image_dimension_count)
    {
        throw ngraph_error("Convolution filter input must have rank of 2 + n_image_dimensions.");
    }

    m_output_channel_count = filters_shape[0];
    if (m_output_channel_count == 0)
    {
        throw ngraph_error("Convolution requires at least one output channel.");
    }

    if (filters_shape[1] != m_input_channel_count)
    {
        throw ngraph_error("Convolution image batch and filter input channel counts do not match.");
    }

    //
    // Make sure window movement strides, window dilation strides, and image dilation strides
    // have same rank as Di.
    //
    if (m_window_movement_strides.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Convolution window movement stride rank does not match number of image dimensions.");
    }

    if (m_window_dilation_strides.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Convolution window dilation stride rank does not match number of image dimensions.");
    }

    if (m_image_dilation_strides.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Convolution image dilation stride rank does not match number of image dimensions.");
    }

    //
    // Make sure padding-below and padding-above shapes have same rank as Di.
    //
    if (m_padding_below.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Convolution padding-below rank does not match number of image dimensions.");
    }

    if (m_padding_above.size() != m_image_dimension_count)
    {
        throw ngraph_error(
            "Convolution padding-above rank does not match number of image dimensions.");
    }

    //
    // Extract input image shape Di and make sure all dimensions are larger than 0 after padding and dilation.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (image_dilation_strides[i] == 0)
        {
            throw ngraph_error("Convolution image dilation stride is zero.");
        }

        size_t dim_size = image_batch_shape[1 + 1 + i];
        m_input_image_physical_shape.push_back(dim_size);
        size_t dilated_dim_size = (dim_size - 1) * image_dilation_strides[i] + 1;
        size_t padded_dilated_dim_size = padding_below[i] + dilated_dim_size + padding_above[i];
        m_input_image_virtual_shape.push_back(padded_dilated_dim_size);

        if (m_input_image_virtual_shape[i] == 0)
        {
            throw ngraph_error(
                "Convolution input image dimension after dilation is zero even with padding.");
        }
    }

    //
    // Extract the physical shape Wp of the convolution window, *not* including dilation, from the filter dimensions.
    // At the same time, make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        m_window_physical_shape.push_back(filters_shape[1 + 1 + i]);
        if (m_window_physical_shape[i] == 0)
        {
            throw ngraph_error("Convolution window shape has a zero-length axis.");
        }
    }

    //
    // Compute physical shape Wp of the convolution window, *including* dilation. At the same time, make sure all
    // window dilation strides are larger than 0, and that the dilated filter fits within the image dimensions.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (m_window_dilation_strides[i] == 0)
        {
            throw ngraph_error("Convolution window axis dilation stride is zero.");
        }

        m_window_virtual_shape.push_back(
            (m_window_physical_shape[i] - 1) * m_window_dilation_strides[i] + 1);

        if (m_window_virtual_shape[i] > m_input_image_virtual_shape[i])
        {
            throw ngraph_error(
                "Convolution window after dilation is larger than the image even with padding.");
        }
    }

    //
    // Compute image output shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    for (size_t i = 0; i < m_image_dimension_count; i++)
    {
        if (m_window_movement_strides[i] == 0)
        {
            throw ngraph_error("Convolution window axis movement stride is zero.");
        }
        m_output_image_shape.push_back(
            ceil_div(m_input_image_virtual_shape[i] - m_window_virtual_shape[i] + 1,
                     m_window_movement_strides[i]));
    }

    //
    // Construct result shape: NCoDo.
    //
    Shape result_shape(1 + 1 + m_image_dimension_count);
    result_shape[0] = m_batch_size;
    result_shape[1] = m_output_channel_count;
    std::copy(m_output_image_shape.begin(), m_output_image_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(get_inputs().at(0).get_element_type(), result_shape);
}

Strides op::Convolution::default_strides(const std::shared_ptr<Node>& image_batch)
{
    auto& image_batch_shape = image_batch->get_shape();
    if (image_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }
    return Strides(image_batch_shape.size() - 2, 1);
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const Shape& padding_below,
                             const Shape& padding_above)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  default_strides(image_batch))
{
}

Shape op::Convolution::default_padding(const std::shared_ptr<Node>& image_batch)
{
    auto& image_batch_shape = image_batch->get_shape();
    if (image_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }
    return Shape(image_batch_shape.size() - 2, 0);
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  default_strides(image_batch),
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters)
    : Convolution(image_batch,
                  filters,
                  default_strides(image_batch),
                  default_strides(image_batch),
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

std::shared_ptr<Node>
    op::Convolution::copy_with_new_args(const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Convolution>(new_args.at(0),
                                         new_args.at(1),
                                         m_window_movement_strides,
                                         m_window_dilation_strides,
                                         m_padding_below,
                                         m_padding_above,
                                         m_image_dilation_strides);
}

/*
void op::Convolution::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
}
*/
