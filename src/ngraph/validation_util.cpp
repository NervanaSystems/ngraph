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

#include "ngraph/validation_util.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

//
// Infers the spatial shape of a single item in a convolution batch.
//
Shape ngraph::infer_windowed_reduction_output_shape(const Node* node,
                                                    const Shape& data_shape,
                                                    const Strides& data_dilation,
                                                    const CoordinateDiff& data_padding_below,
                                                    const CoordinateDiff& data_padding_above,
                                                    const Shape& window_shape,
                                                    const Strides& window_strides,
                                                    const Strides& window_dilation,
                                                    bool is_window_all_in_padding_allowed)
{
    NODE_VALIDATION_ASSERT(node, data_shape.size() == data_dilation.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the data dilation (" << data_dilation << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == data_padding_below.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the data padding below (" << data_padding_below << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == data_padding_above.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the data padding above (" << data_padding_above << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == window_shape.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the window shape (" << window_shape << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == window_strides.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the window strides (" << window_strides << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == window_dilation.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the window dilation (" << window_dilation << ").";

    Shape output_shape(data_shape.size());

    for (size_t i = 0; i < data_shape.size(); i++)
    {
        NODE_VALIDATION_ASSERT(node, data_shape[i] > 0)
            << "Data shape (" << data_shape << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, data_dilation[i] > 0)
            << "Data dilation (" << data_dilation << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_shape[i] > 0)
            << "Window shape (" << window_shape << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_shape[i] <= data_shape[i])
            << "Window shape (" << window_shape << ") is smaller than data shape (" << data_shape
            << ") at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_strides[i] > 0)
            << "Window strides (" << window_strides << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_dilation[i] > 0)
            << "Window dilation (" << window_dilation << ") has zero dimension at axis " << i
            << ".";

        ptrdiff_t data_padded_dilated_dim = ptrdiff_t(data_dilation[i] * (data_shape[i] - 1) + 1) +
                                            data_padding_below[i] + data_padding_above[i];
        size_t window_dilated_dim = window_dilation[i] * (window_shape[i] - 1) + 1;

        NODE_VALIDATION_ASSERT(node, data_padded_dilated_dim > 0)
            << "Data shape after padding and dilation has dimension less than 1 (dim: "
            << data_padded_dilated_dim << ") at dimension " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_dilated_dim > 0)
            << "Window after dilation has dimension less than 1 (dim: " << window_dilated_dim
            << ") at dimension " << i << ".";
        NODE_VALIDATION_ASSERT(node, window_dilated_dim <= size_t(data_padded_dilated_dim))
            << "Window after dilation has dimension (dim: " << window_dilated_dim
            << ") larger than the data shape after padding (dim: " << data_padded_dilated_dim
            << ") at dimension " << i << ".";
        NODE_VALIDATION_ASSERT(node,
                               is_window_all_in_padding_allowed ||
                                   (window_dilated_dim >= data_padding_below[i] &&
                                    window_dilated_dim >= data_padding_above[i]))
            << "Window after dilation is sometimes entirely in the padding area for axis " << i
            << "(dilated window dimension: " << window_dilated_dim
            << ", padding below dimension: " << data_padding_below[i]
            << ", padding above dimension: " << data_padding_above[i] << ") and this is not "
            << "allowed.";

        size_t output_dim =
            ceil_div(size_t(data_padded_dilated_dim) - window_dilated_dim + 1, window_strides[i]);
        output_shape[i] = output_dim;
    }

    return output_shape;
}

//
// Infers the output batch shape and element type for convolution fprop.
//
std::tuple<element::Type, Shape>
    ngraph::infer_convolution_forward(const Node* node,
                                      element::Type et_batch,
                                      element::Type et_filters,
                                      const Shape& data_batch_shape,
                                      const Strides& data_dilation,
                                      const CoordinateDiff& data_padding_below,
                                      const CoordinateDiff& data_padding_above,
                                      const Shape& filters_shape,
                                      const Strides& filter_strides,
                                      const Strides& filter_dilation)
{
    NODE_VALIDATION_ASSERT(node, et_batch == et_filters)
        << "Element types for data batch and filters do not match (data batch element type: "
        << et_batch << ", filters element type: " << et_filters << ").";

    NODE_VALIDATION_ASSERT(node, data_batch_shape.size() >= 3)
        << "Data batch must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    NODE_VALIDATION_ASSERT(node, filters_shape.size() >= 3)
        << "Filters must have rank of at least 3 (one output-channel axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(filters shape: " << filters_shape << ").";

    size_t batch_size = data_batch_shape[0];
    size_t data_channel_count = data_batch_shape[1];
    Shape data_spatial_shape(data_batch_shape.begin() + 2, data_batch_shape.end());

    size_t filter_output_channel_count = filters_shape[0];
    size_t filter_input_channel_count = filters_shape[1];
    Shape filter_spatial_shape(filters_shape.begin() + 2, filters_shape.end());

    NODE_VALIDATION_ASSERT(node, batch_size > 0) << "Batch size is zero.";

    NODE_VALIDATION_ASSERT(node, data_channel_count > 0) << "Data batch channel count is zero.";

    NODE_VALIDATION_ASSERT(node, data_channel_count == filter_input_channel_count)
        << "Data batch channel count (" << data_channel_count << ") does not match filter input "
        << "channel count (" << filter_input_channel_count << ").";

    NODE_VALIDATION_ASSERT(node, filter_output_channel_count > 0)
        << "Filter output channel count is zero.";

    Shape data_output_shape = infer_windowed_reduction_output_shape(node,
                                                                    data_spatial_shape,
                                                                    data_dilation,
                                                                    data_padding_below,
                                                                    data_padding_above,
                                                                    filter_spatial_shape,
                                                                    filter_strides,
                                                                    filter_dilation,
                                                                    true);

    Shape batch_output_shape(data_batch_shape.size());
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = filter_output_channel_count;
    std::copy(data_output_shape.begin(), data_output_shape.end(), batch_output_shape.begin() + 2);

    return std::make_tuple(et_batch, batch_output_shape);
}

//
// Infers the output batch shape and element type for batched pooling fprop.
//
Shape ngraph::infer_batched_pooling_forward(const Node* node,
                                            const Shape& data_batch_shape,
                                            const CoordinateDiff& data_padding_below,
                                            const CoordinateDiff& data_padding_above,
                                            const Shape& window_shape,
                                            const Strides& window_strides,
                                            bool is_window_all_in_padding_allowed)
{
    NODE_VALIDATION_ASSERT(node, data_batch_shape.size() >= 3)
        << "Data batch must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    size_t spatial_dimension_count = data_batch_shape.size() - 2;

    NODE_VALIDATION_ASSERT(node, data_padding_below.size() == spatial_dimension_count)
        << "Data padding below (" << data_padding_below << ") does not have required rank ("
        << spatial_dimension_count << ").";

    NODE_VALIDATION_ASSERT(node, data_padding_above.size() == spatial_dimension_count)
        << "Data padding above (" << data_padding_above << ") does not have required rank ("
        << spatial_dimension_count << ").";

    NODE_VALIDATION_ASSERT(node, window_shape.size() == spatial_dimension_count)
        << "Window shape (" << window_shape << ") does not have required rank ("
        << spatial_dimension_count << ").";

    NODE_VALIDATION_ASSERT(node, window_strides.size() == spatial_dimension_count)
        << "Window shape (" << window_strides << ") does not have required rank ("
        << spatial_dimension_count << ").";

    size_t batch_size = data_batch_shape[0];
    size_t channel_count = data_batch_shape[1];
    Shape data_spatial_shape(data_batch_shape.begin() + 2, data_batch_shape.end());

    NODE_VALIDATION_ASSERT(node, batch_size > 0) << "Batch size is zero.";

    NODE_VALIDATION_ASSERT(node, channel_count > 0) << "Channel count is zero.";

    // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
    Strides data_dilation(spatial_dimension_count, 1);
    Strides window_dilation(spatial_dimension_count, 1);

    Shape data_output_shape =
        infer_windowed_reduction_output_shape(node,
                                              data_spatial_shape,
                                              data_dilation,
                                              data_padding_below,
                                              data_padding_above,
                                              window_shape,
                                              window_strides,
                                              window_dilation,
                                              is_window_all_in_padding_allowed);

    Shape batch_output_shape(data_batch_shape.size());
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = channel_count;
    std::copy(data_output_shape.begin(), data_output_shape.end(), batch_output_shape.begin() + 2);

    return batch_output_shape;
}
