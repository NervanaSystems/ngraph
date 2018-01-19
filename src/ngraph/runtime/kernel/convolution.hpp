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

#pragma once

#include <cmath>

#include "ngraph/common.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void convolution(T* arg0,
                             T* arg1,
                             T* out,
                             const Shape& arg0_shape,
                             const Shape& arg1_shape,
                             const Shape& out_shape,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const Padding& padding_below,
                             const Padding& padding_above,
                             const Strides& image_dilation_strides)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (Coordinate out_coord : output_transform)
                {
                    // Our output coordinate O will have the form:
                    //
                    //   (img,chan_out,i_1,...,i_n)

                    size_t img_index = out_coord[0];
                    size_t output_channel = out_coord[1];

                    // For the input images we need to iterate the coordinate:
                    //
                    //   I:
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (img,0,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                    //
                    //     (img+1,chans_in_count,s_1*i_1 + l_1*filter_dims_1,...,s_n*i_n + l_n*filter_dims_n)
                    //
                    // with strides:
                    //
                    //   (1,l_1,...,l_n).
                    //
                    // Note that we are iterating within the *padded* and *dilated* image batch, so further
                    // down we must check the current coordinate is in the padding or dilation gap.

                    size_t n_image_dimensions = arg0_shape.size() - 2;
                    size_t n_input_channels = arg0_shape[1];

                    Coordinate input_batch_transform_start(2 + n_image_dimensions);
                    Coordinate input_batch_transform_end(2 + n_image_dimensions);
                    Strides input_batch_transform_movement_strides(2 + n_image_dimensions, 1);
                    Padding input_batch_transform_padding_below(2 + n_image_dimensions, 0);
                    Padding input_batch_transform_padding_above(2 + n_image_dimensions, 0);
                    Strides input_batch_transform_dilation_strides(2 + n_image_dimensions, 1);

                    input_batch_transform_start[0] = img_index;
                    input_batch_transform_end[0] = img_index + 1;
                    input_batch_transform_start[1] = 0;
                    input_batch_transform_end[1] = n_input_channels;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        size_t window_dilation_stride = window_dilation_strides[i - 2];
                        size_t window_movement_stride = window_movement_strides[i - 2];
                        ssize_t below_pad = padding_below[i - 2];
                        ssize_t above_pad = padding_above[i - 2];
                        size_t image_dilation_stride = image_dilation_strides[i - 2];

                        input_batch_transform_start[i] = window_movement_stride * out_coord[i];
                        input_batch_transform_end[i] =
                            input_batch_transform_start[i] +
                            (arg1_shape[i] - 1) * window_dilation_stride + 1;
                        input_batch_transform_movement_strides[i] = window_dilation_stride;
                        input_batch_transform_padding_below[i] = below_pad;
                        input_batch_transform_padding_above[i] = above_pad;
                        input_batch_transform_dilation_strides[i] = image_dilation_stride;
                    }

                    AxisVector input_batch_transform_axis_order(2 + n_image_dimensions);
                    size_t n = 0;
                    std::generate(input_batch_transform_axis_order.begin(),
                                  input_batch_transform_axis_order.end(),
                                  [&n]() -> size_t { return n++; });

                    CoordinateTransform input_batch_transform(
                        arg0_shape,
                        input_batch_transform_start,
                        input_batch_transform_end,
                        input_batch_transform_movement_strides,
                        input_batch_transform_axis_order,
                        input_batch_transform_padding_below,
                        input_batch_transform_padding_above,
                        input_batch_transform_dilation_strides);

                    // Simultaneously with iterating I, for the filters we need to iterate the coordinate:
                    //
                    //   F
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (chan_out,0,0,...,0) -> (chan_out+1,chans_in_count,filter_dims_1,...,filter_dims_n)
                    //
                    // with unit stride.

                    Shape filter_transform_start(2 + n_image_dimensions);
                    Shape filter_transform_end(2 + n_image_dimensions);

                    filter_transform_start[0] = output_channel;
                    filter_transform_end[0] = output_channel + 1;
                    filter_transform_start[1] = 0;
                    filter_transform_end[1] = n_input_channels;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        filter_transform_start[i] = 0;
                        filter_transform_end[i] = arg1_shape[i];
                    }

                    CoordinateTransform filter_transform(
                        arg1_shape, filter_transform_start, filter_transform_end);

                    // As we go, we sum up:
                    //
                    //   output[O] += arg0[I] * arg1[F].

                    T result = 0;

                    CoordinateTransform::Iterator input_it = input_batch_transform.begin();
                    CoordinateTransform::Iterator filter_it = filter_transform.begin();

                    while (input_it != input_batch_transform.end() &&
                           filter_it != filter_transform.end())
                    {
                        const Coordinate& input_batch_coord = *input_it;
                        const Coordinate& filter_coord = *filter_it;

                        T v = input_batch_transform.has_source_coordinate(input_batch_coord)
                                  ? arg0[input_batch_transform.index(input_batch_coord)]
                                  : 0;

                        result += v * arg1[filter_transform.index(filter_coord)];

                        ++input_it;
                        ++filter_it;
                    }

                    out[output_transform.index(out_coord)] = result;
                }
            }
        }
    }
}
