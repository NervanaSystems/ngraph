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

#pragma once

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void general_convolution(const T* data,
                                     const T* filters,
                                     T* out,
                                     const Shape& data_shape,
                                     const Shape& filters_shape,
                                     const Shape& out_shape,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     size_t data_batch_axis,
                                     size_t data_channel_axis,
                                     size_t filters_data_channel_axis,
                                     size_t filters_out_channel_axis,
                                     size_t out_batch_axis,
                                     size_t out_channel_axis,
                                     bool rotate_filter)
            {
                // Comments throughout assume without loss of generality that:
                //
                // * batch axes for both data and output are 0
                // * data channel axes for both data and filters are 1
                // * output channel axes for filters is 0
                // * output channel axis for output is 1
                // * rotate_filter is false

                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate O will have the form:
                    //
                    //   (N,chan_out,i_1,...,i_n)

                    size_t batch_index = out_coord[out_batch_axis];
                    size_t output_channel = out_coord[out_channel_axis];

                    // For the data we need to iterate the coordinate:
                    //
                    //   I:
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (N,0,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                    //
                    //     (N+1,chans_in_count,s_1*i_1 + l_1*filter_dims_1,...,s_n*i_n + l_n*filter_dims_n)
                    //
                    // with strides:
                    //
                    //   (1,l_1,...,l_n).
                    //
                    // Note that we are iterating within the *padded* and *dilated* data batch, so further
                    // down we must check the current coordinate is in the padding or dilation gap.

                    size_t n_spatial_dimensions = data_shape.size() - 2;
                    size_t n_data_channels = data_shape[data_channel_axis];

                    Coordinate data_transform_start(2 + n_spatial_dimensions);
                    Coordinate data_transform_end(2 + n_spatial_dimensions);
                    Strides data_transform_movement_strides(2 + n_spatial_dimensions, 1);
                    CoordinateDiff data_transform_padding_below(2 + n_spatial_dimensions, 0);
                    CoordinateDiff data_transform_padding_above(2 + n_spatial_dimensions, 0);
                    Strides data_transform_dilation_strides(2 + n_spatial_dimensions, 1);

                    data_transform_start[data_batch_axis] = batch_index;
                    data_transform_end[data_batch_axis] = batch_index + 1;
                    data_transform_start[data_channel_axis] = 0;
                    data_transform_end[data_channel_axis] = n_data_channels;

                    for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
                    {
                        size_t window_dilation_stride = window_dilation_strides[i - 2];
                        size_t window_movement_stride = window_movement_strides[i - 2];
                        std::ptrdiff_t below_pad = padding_below[i - 2];
                        std::ptrdiff_t above_pad = padding_above[i - 2];
                        size_t data_dilation_stride = data_dilation_strides[i - 2];

                        data_transform_start[i] = window_movement_stride * out_coord[i];
                        data_transform_end[i] = data_transform_start[i] +
                                                (filters_shape[i] - 1) * window_dilation_stride + 1;
                        data_transform_movement_strides[i] = window_dilation_stride;
                        data_transform_padding_below[i] = below_pad;
                        data_transform_padding_above[i] = above_pad;
                        data_transform_dilation_strides[i] = data_dilation_stride;
                    }

                    AxisVector data_transform_axis_order(2 + n_spatial_dimensions);
                    for (size_t i = 0; i < data_transform_axis_order.size(); i++)
                    {
                        data_transform_axis_order[i] = i;
                    }

                    CoordinateTransform data_transform(data_shape,
                                                       data_transform_start,
                                                       data_transform_end,
                                                       data_transform_movement_strides,
                                                       data_transform_axis_order,
                                                       data_transform_padding_below,
                                                       data_transform_padding_above,
                                                       data_transform_dilation_strides);

                    // Simultaneously with iterating I, for the filters we need to iterate the coordinate:
                    //
                    //   F
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (chan_out,0,0,...,0) -> (chan_out+1,chans_in_count,filter_dims_1,...,filter_dims_n)
                    //
                    // with unit stride.

                    Shape filter_transform_start(2 + n_spatial_dimensions);
                    Shape filter_transform_end(2 + n_spatial_dimensions);

                    filter_transform_start[filters_out_channel_axis] = output_channel;
                    filter_transform_end[filters_out_channel_axis] = output_channel + 1;
                    filter_transform_start[filters_data_channel_axis] = 0;
                    filter_transform_end[filters_data_channel_axis] = n_data_channels;

                    for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
                    {
                        filter_transform_start[i] = 0;
                        filter_transform_end[i] = filters_shape[i];
                    }

                    CoordinateTransform filter_transform(
                        filters_shape, filter_transform_start, filter_transform_end);

                    // As we go, we sum up:
                    //
                    //   output[O] += data[I] * filters[F].

                    T result = 0;

                    CoordinateTransform::Iterator data_it = data_transform.begin();
                    CoordinateTransform::Iterator filter_it = filter_transform.begin();
                    CoordinateTransform::Iterator data_it_end = data_transform.end();
                    CoordinateTransform::Iterator filter_it_end = filter_transform.end();

                    while (data_it != data_it_end && filter_it != filter_it_end)
                    {
                        const Coordinate& data_coord = *data_it;
                        Coordinate filter_coord = *filter_it;

                        if (rotate_filter)
                        {
                            Shape target_shape = filter_transform.get_target_shape();

                            // Note that we only reverse the spatial dimensions here (loop
                            // starts at 2)
                            for (size_t i = 2; i < filter_coord.size(); i++)
                            {
                                filter_coord[i] = target_shape[i] - filter_coord[i] - 1;
                            }
                        }

                        T v = data_transform.has_source_coordinate(data_coord)
                                  ? data[data_transform.index(data_coord)]
                                  : 0;

                        result += v * filters[filter_transform.index(filter_coord)];

                        ++data_it;
                        ++filter_it;
                    }

                    out[output_transform.index(out_coord)] = result;
                }
            }

            template <typename T>
            void convolution(const T* data,
                             const T* filters,
                             T* out,
                             const Shape& data_shape,
                             const Shape& filters_shape,
                             const Shape& out_shape,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above,
                             const Strides& data_dilation_strides)
            {
                general_convolution(data,
                                    filters,
                                    out,
                                    data_shape,
                                    filters_shape,
                                    out_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides,
                                    0,
                                    1,
                                    1,
                                    0,
                                    0,
                                    1,
                                    false);
            }

            template <typename T>
            void convolution_backprop_filters(const T* data,
                                              const T* output_delta,
                                              T* out,
                                              const Shape& filters_shape,
                                              const Shape& data_shape,
                                              const Shape& output_delta_shape,
                                              const Shape& out_shape,
                                              const Strides& window_dilation_strides,
                                              const Strides& window_movement_strides,
                                              const CoordinateDiff& padding_below,
                                              const CoordinateDiff& padding_above,
                                              const Strides& data_dilation_strides)
            {
                size_t spatial_dim_count = static_cast<size_t>(output_delta_shape.size()) - 2;
                CoordinateDiff padding_above_backward;
                padding_above_backward.resize(spatial_dim_count);

                for (size_t i = 0; i < spatial_dim_count; i++)
                {
                    padding_above_backward[i] =
                        padding_above[i] -
                        (padding_below[i] +
                         (static_cast<ptrdiff_t>(data_shape[i + 2]) - 1) *
                             data_dilation_strides[i] +
                         padding_above[i] -
                         (filters_shape[i + 2] - 1) * window_dilation_strides[i]) %
                            window_movement_strides[i];
                }

                general_convolution(data,
                                    output_delta,
                                    out,
                                    data_shape,
                                    output_delta_shape,
                                    out_shape,
                                    window_dilation_strides,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above_backward,
                                    data_dilation_strides,
                                    1,
                                    0,
                                    0,
                                    1,
                                    1,
                                    0,
                                    false);
            }

            template <typename T>
            void convolution_backprop_data(const T* output_delta,
                                           const T* filters,
                                           T* out,
                                           const Shape& data_shape,
                                           const Shape& output_delta_shape,
                                           const Shape& filters_shape,
                                           const Shape& out_shape,
                                           const Strides& data_dilation_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& window_movement_strides)
            {
                size_t spatial_dim_count = static_cast<size_t>(data_shape.size()) - 2;

                CoordinateDiff padding_below_backward;
                padding_below_backward.resize(spatial_dim_count);
                CoordinateDiff padding_above_backward;
                padding_above_backward.resize(spatial_dim_count);

                for (size_t i = 0; i < spatial_dim_count; i++)
                {
                    padding_below_backward[i] = (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                                                    window_dilation_strides[i] -
                                                padding_below[i];
                    padding_above_backward[i] =
                        (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                            window_dilation_strides[i] +
                        ((padding_below[i] + ((data_shape[i + 2]) - 1) * data_dilation_strides[i] +
                          padding_above[i] -
                          (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                              window_dilation_strides[i]) %
                         window_movement_strides[i]) -
                        padding_above[i];
                }
                general_convolution(output_delta,
                                    filters,
                                    out,
                                    output_delta_shape,
                                    filters_shape,
                                    out_shape,
                                    data_dilation_strides,
                                    window_dilation_strides,
                                    padding_below_backward,
                                    padding_above_backward,
                                    window_movement_strides,
                                    0,
                                    1,
                                    0,
                                    1,
                                    0,
                                    1,
                                    true);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
