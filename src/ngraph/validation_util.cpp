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
// Infers the output shape of a windowed reduction operation, where the data may be dilated and/or
// padded, and the reduction window may be strided and/or dilated.
//
// TODO(amprocte): The messages here would be a bit friendlier if we didn't say "after
// padding/after dilation" for cases where there is actually no padding/dilation.
//
PartialShape ngraph::infer_windowed_reduction_output_shape(const Node* node,
                                                           const PartialShape& data_shape,
                                                           const Strides& data_dilation,
                                                           const CoordinateDiff& data_padding_below,
                                                           const CoordinateDiff& data_padding_above,
                                                           const PartialShape& window_shape,
                                                           const Strides& window_strides,
                                                           const Strides& window_dilation,
                                                           bool is_window_all_in_padding_allowed)
{
    PartialShape data_shape_merged{PartialShape::dynamic()};

    NODE_VALIDATION_ASSERT(node,
                           data_shape_merged.merge_rank(data_shape.rank()) &&
                               data_shape_merged.merge_rank(data_dilation.size()) &&
                               data_shape_merged.merge_rank(data_padding_below.size()) &&
                               data_shape_merged.merge_rank(data_padding_above.size()) &&
                               data_shape_merged.merge_rank(window_shape.rank()) &&
                               data_shape_merged.merge_rank(window_strides.size()) &&
                               data_shape_merged.merge_rank(window_dilation.size()))
        << "Ranks for data shape (" << data_shape << "), data dilation (" << data_dilation
        << "), padding below (" << data_padding_below << "), padding above (" << data_padding_above
        << "), window shape (" << window_shape << "), window strides (" << window_strides
        << "), and window dilation (" << window_dilation << ") do not match.";

    PartialShape output_shape = PartialShape::dynamic(data_shape_merged.rank());

    if (output_shape.rank().is_static())
    {
        for (size_t i = 0; i < static_cast<size_t>(output_shape.rank()); i++)
        {
            NODE_VALIDATION_ASSERT(node, data_dilation[i] > 0)
                << "Data dilation (" << data_dilation << ") has zero dimension at axis " << i
                << ".";
            NODE_VALIDATION_ASSERT(node, window_strides[i] > 0)
                << "Window strides (" << window_strides << ") has zero dimension at axis " << i
                << ".";
            NODE_VALIDATION_ASSERT(node, window_dilation[i] > 0)
                << "Window dilation (" << window_dilation << ") has zero dimension at axis " << i
                << ".";

            bool data_dim_static = data_shape.rank().is_static() && data_shape[i].is_static();
            bool window_dim_static = window_shape.rank().is_static() && window_shape[i].is_static();

            ptrdiff_t data_padded_dilated_dim = -1;
            if (data_dim_static)
            {
                data_padded_dilated_dim = (static_cast<ptrdiff_t>(data_dilation[i]) *
                                           (static_cast<ptrdiff_t>(data_shape[i]) - 1)) +
                                          1 + data_padding_below[i] + data_padding_above[i];
                NODE_VALIDATION_ASSERT(node, data_padded_dilated_dim > 0)
                    << "Data shape after padding and dilation has dimension less than 1 (dim: "
                    << data_padded_dilated_dim << ") at axis " << i << ".";
            }

            ptrdiff_t window_dilated_dim = -1;
            if (window_dim_static)
            {
                window_dilated_dim = static_cast<ptrdiff_t>(window_dilation[i]) *
                                         (static_cast<ptrdiff_t>(window_shape[i]) - 1) +
                                     1;

                NODE_VALIDATION_ASSERT(node, window_dilated_dim > 0)
                    << "Window after dilation has dimension less than 1 (dim: "
                    << window_dilated_dim << ") at axis " << i << ".";

                NODE_VALIDATION_ASSERT(node,
                                       is_window_all_in_padding_allowed ||
                                           (window_dilated_dim > data_padding_below[i] &&
                                            window_dilated_dim > data_padding_above[i]))
                    << "Window after dilation is sometimes entirely in the padding area for axis "
                    << i << " (dilated window dimension: " << window_dilated_dim
                    << ", padding below dimension: " << data_padding_below[i]
                    << ", padding above dimension: " << data_padding_above[i]
                    << ") and this is not "
                    << "allowed.";
            }

            if (data_dim_static && window_dim_static)
            {
                NODE_VALIDATION_ASSERT(node, window_dilated_dim <= data_padded_dilated_dim)
                    << "Window after dilation has dimension (dim: " << window_dilated_dim
                    << ") larger than the data shape after padding (dim: "
                    << data_padded_dilated_dim << ") at axis " << i << ".";

                output_shape[i] = ceil_div(static_cast<size_t>(data_padded_dilated_dim) -
                                               static_cast<size_t>(window_dilated_dim) + 1,
                                           window_strides[i]);
            }
        }
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
                                                                    true)
                                  .to_shape();

    Shape batch_output_shape(data_batch_shape.size());
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = filter_output_channel_count;
    std::copy(data_output_shape.begin(), data_output_shape.end(), batch_output_shape.begin() + 2);

    return std::make_tuple(et_batch, batch_output_shape);
}

//
// Infers the output batch shape and element type for batched pooling fprop.
//
PartialShape ngraph::infer_batched_pooling_forward(const Node* node,
                                                   const PartialShape& data_batch_shape,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   bool is_window_all_in_padding_allowed)
{
    NODE_VALIDATION_ASSERT(node,
                           data_batch_shape.rank().is_dynamic() ||
                               static_cast<size_t>(data_batch_shape.rank()) >= 3)
        << "Data batch must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    PartialShape data_spatial_shape{PartialShape::dynamic()};

    NODE_VALIDATION_ASSERT(node,
                           data_spatial_shape.merge_rank(data_batch_shape.rank() - 2) &&
                               data_spatial_shape.merge_rank(data_padding_below.size()) &&
                               data_spatial_shape.merge_rank(data_padding_above.size()) &&
                               data_spatial_shape.merge_rank(window_shape.rank()) &&
                               data_spatial_shape.merge_rank(window_strides.size()))
        << "Ranks for data item shape (data batch has shape " << data_batch_shape
        << ", so data item rank is " << (data_batch_shape.rank() - 2) << "), padding below ("
        << data_padding_below << "), padding above (" << data_padding_above << "), window shape ("
        << window_shape << "), and window strides (" << window_strides << ") do not match.";

    Dimension batch_size{Dimension::dynamic()};
    Dimension channel_count{Dimension::dynamic()};
    PartialShape data_output_spatial_shape{PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static())
    {
        batch_size = data_batch_shape[0];
        channel_count = data_batch_shape[1];

        for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
        {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        NODE_VALIDATION_ASSERT(node, batch_size.is_dynamic() || static_cast<size_t>(batch_size) > 0)
            << "Batch size is zero.";

        NODE_VALIDATION_ASSERT(node,
                               channel_count.is_dynamic() || static_cast<size_t>(channel_count) > 0)
            << "Channel count is zero.";

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        Strides data_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);
        Strides window_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);

        data_output_spatial_shape =
            infer_windowed_reduction_output_shape(node,
                                                  data_spatial_shape,
                                                  data_dilation,
                                                  data_padding_below,
                                                  data_padding_above,
                                                  window_shape,
                                                  window_strides,
                                                  window_dilation,
                                                  is_window_all_in_padding_allowed);
    }

    PartialShape data_batch_output_shape{
        PartialShape::dynamic(data_output_spatial_shape.rank() + 2)};
    data_batch_output_shape[0] = batch_size;
    data_batch_output_shape[1] = channel_count;

    for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
    {
        data_batch_output_shape[i + 2] = data_output_spatial_shape[i];
    }

    return data_batch_output_shape;
}
