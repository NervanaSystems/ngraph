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

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void max_pool(T* arg,
                          T* out,
                          const Shape& arg_shape,
                          const Shape& out_shape,
                          const Shape& window_shape,
                          const Strides& window_movement_strides)
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

                    size_t n_image_dimensions = arg_shape.size() - 2;

                    Coordinate input_batch_transform_start(2 + n_image_dimensions);
                    Coordinate input_batch_transform_end(2 + n_image_dimensions);

                    input_batch_transform_start[0] = img_index;
                    input_batch_transform_end[0] = img_index + 1;
                    input_batch_transform_start[1] = channel;
                    input_batch_transform_end[1] = channel + 1;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        size_t window_shape_this_dim = window_shape[i - 2];
                        size_t movement_stride = window_movement_strides[i - 2];

                        input_batch_transform_start[i] = movement_stride * out_coord[i];
                        input_batch_transform_end[i] =
                            input_batch_transform_start[i] + window_shape_this_dim;
                    }

                    CoordinateTransform input_batch_transform(
                        arg_shape, input_batch_transform_start, input_batch_transform_end);

                    // As we go, we compute the maximum value:
                    //
                    //   output[O] = max(output[O],arg[I])

                    T result = std::numeric_limits<T>::has_infinity
                                   ? -std::numeric_limits<T>::infinity()
                                   : std::numeric_limits<T>::min();

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        T x = arg[input_batch_transform.index(input_batch_coord)];
                        result = x > result ? x : result;
                    }

                    out[output_transform.index(out_coord)] = result;
                }
            }
        }
    }
}
