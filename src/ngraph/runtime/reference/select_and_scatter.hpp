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
            void select_and_scatter(const T* arg_selectee,
                                    const T* arg_source,
                                    const T* arg_init,
                                    T* out,
                                    const Shape& arg_selectee_shape,
                                    const Shape& arg_source_shape,
                                    const Shape& out_shape,
                                    std::function<char(T, T)> selection_function,
                                    std::function<T(T, T)> scatter_function,
                                    const Shape& window_shape,
                                    const Strides& window_movement_strides)
            {
                // First write every element of the output with the supplied initial value.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    out[output_transform.index(out_coord)] = *arg_init;
                }

                // Slide the window over selectee/output.
                Shape window_start_corner_transform_start(arg_selectee_shape.size(), 0);
                Shape window_start_corner_transform_end(arg_selectee_shape.size());

                for (size_t i = 0; i < arg_selectee_shape.size(); i++)
                {
                    window_start_corner_transform_end[i] =
                        arg_selectee_shape[i] - window_shape[i] + 1;
                }

                CoordinateTransform window_start_corner_transform(
                    arg_selectee_shape,
                    window_start_corner_transform_start,
                    window_start_corner_transform_end,
                    window_movement_strides);

                CoordinateTransform source_transform(arg_source_shape);
                CoordinateTransform::Iterator source_it = source_transform.begin();

                for (Coordinate window_start_coord : window_start_corner_transform)
                {
                    // We need a physical rather than virtual coordinate to start the window.
                    window_start_coord =
                        window_start_corner_transform.to_source_coordinate(window_start_coord);

                    Shape window_transform_end(arg_selectee_shape.size());
                    for (size_t i = 0; i < arg_selectee_shape.size(); i++)
                    {
                        window_transform_end[i] = window_start_coord[i] + window_shape[i];
                    }

                    CoordinateTransform window_transform(
                        arg_selectee_shape, window_start_coord, window_transform_end);

                    bool first_val = true;
                    Coordinate winner_coord;

                    // This initial value is ignored; it's just here so the compiler knows
                    // for sure that winner_val is initialized.
                    T winner_val = 0;

                    for (const Coordinate& challenger_coord : window_transform)
                    {
                        T challenger_val = arg_selectee[window_transform.index(challenger_coord)];

                        if (first_val || selection_function(challenger_val, winner_val))
                        {
                            winner_coord = challenger_coord;
                            winner_val = challenger_val;
                            first_val = false;
                        }
                    }

                    Coordinate source_coord = *source_it;

                    T old_output_val = out[window_transform.index(winner_coord)];
                    T source_val = arg_source[source_transform.index(source_coord)];
                    T new_output_val = scatter_function(old_output_val, source_val);

                    out[window_transform.index(winner_coord)] = new_output_val;

                    ++source_it;
                }
            }
        }
    }
}
