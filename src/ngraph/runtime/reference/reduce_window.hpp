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

#pragma once

#include <cmath>
#include <functional>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reduce_window(const T* arg_reductee,
                               const T* arg_init,
                               T* out,
                               const Shape& arg_reductee_shape,
                               const Shape& out_shape,
                               std::function<T(T, T)> reduction_function,
                               const Shape& window_shape,
                               const Strides& window_movement_strides)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate O will have the form:
                    //
                    //   (i_1,...,i_n)
                    //
                    // For the reductee we need to iterate the coordinate:
                    //
                    //   I:
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                    //
                    //     (s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
                    //
                    // with unit stride.

                    Shape reductee_transform_start;
                    Shape reductee_transform_end;

                    for (size_t i = 0; i < arg_reductee_shape.size(); i++)
                    {
                        size_t window_shape_this_dim = window_shape[i];
                        size_t movement_stride = window_movement_strides[i];

                        reductee_transform_start.push_back(movement_stride * out_coord[i]);
                        reductee_transform_end.push_back(reductee_transform_start[i] +
                                                         window_shape_this_dim);
                    }

                    CoordinateTransform reductee_transform(
                        arg_reductee_shape, reductee_transform_start, reductee_transform_end);

                    // As we go, we compute the reduced value:
                    //
                    //   output[O] := reduction_function(output[O],arg[I])

                    T result = *arg_init;

                    for (const Coordinate& reductee_coord : reductee_transform)
                    {
                        result = reduction_function(
                            result, arg_reductee[reductee_transform.index(reductee_coord)]);
                    }

                    out[output_transform.index(out_coord)] = result;
                }
            }
        }
    }
}
