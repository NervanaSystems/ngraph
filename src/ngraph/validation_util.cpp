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
Shape ngraph::infer_convolution_output_item_shape(const Node* node,
                                                  const Shape& data_shape,
                                                  const Strides& data_dilation,
                                                  const CoordinateDiff& data_padding_below,
                                                  const CoordinateDiff& data_padding_above,
                                                  const Shape& filter_shape,
                                                  const Strides& filter_strides,
                                                  const Strides& filter_dilation)
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

    NODE_VALIDATION_ASSERT(node, data_shape.size() == filter_shape.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the filter shape (" << filter_shape << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == filter_strides.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the filter strides (" << filter_strides << ").";

    NODE_VALIDATION_ASSERT(node, data_shape.size() == filter_dilation.size())
        << "Data shape (" << data_shape << ") does not have same rank as "
        << "the filter dilation (" << filter_dilation << ").";

    Shape output_shape(data_shape.size());

    for (size_t i = 0; i < data_shape.size(); i++)
    {
        NODE_VALIDATION_ASSERT(node, data_shape[i] > 0)
            << "Data shape (" << data_shape << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, data_dilation[i] > 0)
            << "Data dilation (" << data_dilation << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_shape[i] > 0)
            << "Filter shape (" << filter_shape << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_shape[i] <= data_shape[i])
            << "Filter shape (" << filter_shape << ") is smaller than data shape (" << data_shape
            << ") at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_strides[i] > 0)
            << "Filter strides (" << filter_strides << ") has zero dimension at axis " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_dilation[i] > 0)
            << "Filter dilation (" << filter_dilation << ") has zero dimension at axis " << i
            << ".";

        ptrdiff_t data_padded_dilated_dim = ptrdiff_t(data_dilation[i] * (data_shape[i] - 1) + 1) +
                                            data_padding_below[i] + data_padding_above[i];
        size_t filter_dilated_dim = filter_dilation[i] * (filter_shape[i] - 1) + 1;

        NODE_VALIDATION_ASSERT(node, data_padded_dilated_dim > 0)
            << "Data shape after padding and dilation has dimension less than 1 (dim: "
            << data_padded_dilated_dim << ") at dimension " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_dilated_dim > 0)
            << "Filter after dilation has dimension less than 1 (dim: " << filter_dilated_dim
            << ") at dimension " << i << ".";
        NODE_VALIDATION_ASSERT(node, filter_dilated_dim <= size_t(data_padded_dilated_dim))
            << "Filter after dilation has dimension (dim: " << filter_dilated_dim
            << ") larger than the data shape after padding (dim: " << data_padded_dilated_dim
            << ") at dimension " << i << ".";

        size_t output_dim =
            ceil_div(size_t(data_padded_dilated_dim) - filter_dilated_dim + 1, filter_strides[i]);
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

    Shape data_output_shape = infer_convolution_output_item_shape(node,
                                                                  data_spatial_shape,
                                                                  data_dilation,
                                                                  data_padding_below,
                                                                  data_padding_above,
                                                                  filter_spatial_shape,
                                                                  filter_strides,
                                                                  filter_dilation);

    Shape batch_output_shape(data_batch_shape.size());
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = filter_output_channel_count;
    std::copy(data_output_shape.begin(), data_output_shape.end(), batch_output_shape.begin() + 2);

    return std::make_tuple(et_batch, batch_output_shape);
}
