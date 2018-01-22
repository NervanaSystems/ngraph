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

#include <algorithm>
#include <cmath>
#include <vector>

#include "ngraph/common.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void avg_pool_bprop(T* arg,
                                T* delta,
                                T* out, //out is also arg_shape
                                const Shape& arg_shape,
                                const Shape& delta_shape,
                                const Shape& window_shape,
                                const Strides& window_movement_strides,
                                const Shape& padding_below,
                                const Shape& padding_above,
                                bool count_only_physical)
            {
                memset(out, 0, sizeof(T) * shape_size(arg_shape));
                size_t j = 0; //for iterating over delta (ep) elements
                size_t num_elements_in_window = shape_size(window_shape);
                CoordinateTransform output_transform(delta_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    size_t img_index = out_coord[0];
                    size_t channel = out_coord[1];

                    size_t n_image_dimensions = arg_shape.size() - 2;
                    Coordinate input_batch_transform_start(2 + n_image_dimensions);
                    Coordinate input_batch_transform_end(2 + n_image_dimensions);
                    Strides input_batch_transform_source_strides(2 + n_image_dimensions, 1);
                    AxisVector input_batch_transform_source_axis_order(2 + n_image_dimensions);
                    CoordinateDiff input_batch_transform_padding_below(2 + n_image_dimensions);
                    CoordinateDiff input_batch_transform_padding_above(2 + n_image_dimensions);

                    input_batch_transform_start[0] = img_index;
                    input_batch_transform_end[0] = img_index + 1;
                    input_batch_transform_start[1] = channel;
                    input_batch_transform_end[1] = channel + 1;
                    input_batch_transform_padding_below[0] = 0;
                    input_batch_transform_padding_below[1] = 0;
                    input_batch_transform_padding_above[0] = 0;
                    input_batch_transform_padding_above[1] = 0;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        size_t window_shape_this_dim = window_shape[i - 2];
                        size_t movement_stride = window_movement_strides[i - 2];

                        input_batch_transform_start[i] = movement_stride * out_coord[i];
                        input_batch_transform_end[i] =
                            input_batch_transform_start[i] + window_shape_this_dim;
                        input_batch_transform_padding_below[i] = padding_below[i - 2];
                        input_batch_transform_padding_above[i] = padding_above[i - 2];
                    }
                    std::iota(begin(input_batch_transform_source_axis_order),
                              end(input_batch_transform_source_axis_order),
                              0);

                    CoordinateTransform input_batch_transform(
                        arg_shape,
                        input_batch_transform_start,
                        input_batch_transform_end,
                        input_batch_transform_source_strides,
                        input_batch_transform_source_axis_order,
                        input_batch_transform_padding_below,
                        input_batch_transform_padding_above);

                    if (count_only_physical)
                    {
                        num_elements_in_window = 0;
                        //Dumb! But should work for now
                        for (const Coordinate& input_batch_coord : input_batch_transform)
                        {
                            if (input_batch_transform.has_source_coordinate(input_batch_coord))
                            {
                                num_elements_in_window++;
                            }
                        }
                    }

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        if (input_batch_transform.has_source_coordinate(input_batch_coord))
                        {
                            size_t index = input_batch_transform.index(input_batch_coord);
                            out[index] += delta[j] / num_elements_in_window;
                        }
                    }
                    j++; //move to the next ep
                }
            }

            template <typename T>
            void avg_pool(T* arg,
                          T* out,
                          const Shape& arg_shape,
                          const Shape& out_shape,
                          const Shape& window_shape,
                          const Strides& window_movement_strides,
                          const Shape& padding_below,
                          const Shape& padding_above)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate O will have the form:
                    //
                    //   (img,chan,i_1,...,i_n)

                    size_t img_index = out_coord[0];
                    size_t channel = out_coord[1];

                    // For the input images we need to iterate the coordinate:
                    //
                    //   I:
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (img,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                    //
                    //     (img+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
                    //
                    // with unit stride.
                    //
                    // We iterate this over the *padded* image, so below we will need to check for coordinates that fall in the padding area.

                    size_t n_image_dimensions = arg_shape.size() - 2;

                    Coordinate input_batch_transform_start(2 + n_image_dimensions);
                    Coordinate input_batch_transform_end(2 + n_image_dimensions);
                    Strides input_batch_transform_source_strides(2 + n_image_dimensions, 1);
                    AxisVector input_batch_transform_source_axis_order(2 + n_image_dimensions);
                    CoordinateDiff input_batch_transform_padding_below(2 + n_image_dimensions);
                    CoordinateDiff input_batch_transform_padding_above(2 + n_image_dimensions);

                    input_batch_transform_start[0] = img_index;
                    input_batch_transform_end[0] = img_index + 1;
                    input_batch_transform_start[1] = channel;
                    input_batch_transform_end[1] = channel + 1;
                    input_batch_transform_padding_below[0] = 0;
                    input_batch_transform_padding_below[1] = 0;
                    input_batch_transform_padding_above[0] = 0;
                    input_batch_transform_padding_above[1] = 0;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        size_t window_shape_this_dim = window_shape[i - 2];
                        size_t movement_stride = window_movement_strides[i - 2];

                        input_batch_transform_start[i] = movement_stride * out_coord[i];
                        input_batch_transform_end[i] =
                            input_batch_transform_start[i] + window_shape_this_dim;
                        input_batch_transform_padding_below[i] = padding_below[i - 2];
                        input_batch_transform_padding_above[i] = padding_above[i - 2];
                    }

                    for (size_t i = 0; i < arg_shape.size(); i++)
                    {
                        input_batch_transform_source_axis_order[i] = i;
                    }

                    CoordinateTransform input_batch_transform(
                        arg_shape,
                        input_batch_transform_start,
                        input_batch_transform_end,
                        input_batch_transform_source_strides,
                        input_batch_transform_source_axis_order,
                        input_batch_transform_padding_below,
                        input_batch_transform_padding_above);

                    // As we go, we compute the sum value:
                    //
                    //   output[O] := output[O] + arg[I]
                    //
                    // and the number of elements:
                    //
                    //   n_elements := n_elements + 1

                    T result = 0;
                    size_t n_elements = 0;

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        bool in_bounds =
                            input_batch_transform.has_source_coordinate(input_batch_coord);
                        T v = in_bounds ? arg[input_batch_transform.index(input_batch_coord)] : 0;
                        result += v;
                        if (in_bounds)
                        {
                            n_elements++;
                        }
                    }

                    out[output_transform.index(out_coord)] = result / n_elements;
                }
            }
        }
    }
}
