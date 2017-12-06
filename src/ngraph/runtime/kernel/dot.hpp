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
#include <utility>

#include "ngraph/common.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void dot(T* arg0,
                     T* arg1,
                     T* out,
                     const Shape& arg0_shape,
                     const Shape& arg1_shape,
                     const Shape& out_shape,
                     const std::vector<std::pair<size_t, size_t>>& dot_axis_pairs)
            {
                Shape dot_axis_sizes;

                AxisSet arg0_dot_axis_set;
                AxisSet arg1_dot_axis_set;

                // We're going to build two things here:
                //
                //    (1) a pair of sets indicating which axes are being dotted in arg0 and arg1
                //        respectively (arg0_dot_axis_set and arg1_dot_axis_set),
                //    (2) a "pseudo-shape" that holds the lengths of each dot-axis (dot_axis_sizes);
                //        for each dot-axis pair we hold one value, since both axes are the same
                //        length.
                //
                // The order of dot_axis_sizes matches the order of dot_axis_pairs.
                for (std::pair<size_t, size_t> axis_pair : dot_axis_pairs)
                {
                    size_t arg0_axis = axis_pair.first;
                    size_t arg1_axis = axis_pair.second;

                    arg0_dot_axis_set.insert(arg0_axis);
                    arg1_dot_axis_set.insert(arg1_axis);

                    dot_axis_sizes.push_back(arg0_shape[arg0_axis]);
                }

                CoordinateTransform arg0_transform(arg0_shape);
                CoordinateTransform arg1_transform(arg1_shape);
                CoordinateTransform output_transform(out_shape);

                // Create coordinate transforms for arg0 and arg1 that throw away the dotted axes.
                CoordinateTransform arg0_projected_transform(
                    project_shape(arg0_shape, arg0_dot_axis_set));
                CoordinateTransform arg1_projected_transform(
                    project_shape(arg1_shape, arg1_dot_axis_set));

                // Create a coordinate transform that allows us to iterate over all possible values
                // for the dotted axes.
                CoordinateTransform dot_axes_transform(dot_axis_sizes);

                for (Coordinate arg0_projected_coord : arg0_projected_transform)
                {
                    for (Coordinate arg1_projected_coord : arg1_projected_transform)
                    {
                        Coordinate out_coord(arg0_projected_coord.size() +
                                             arg1_projected_coord.size());

                        std::copy(arg0_projected_coord.begin(),
                                  arg0_projected_coord.end(),
                                  out_coord.begin());
                        std::copy(arg1_projected_coord.begin(),
                                  arg1_projected_coord.end(),
                                  out_coord.begin() + arg0_projected_coord.size());

                        size_t out_index = output_transform.index(out_coord);

                        out[out_index] = 0;

                        // Walk along the dotted axes...
                        for (Coordinate dot_axis_positions : dot_axes_transform)
                        {
                            // In order to find the points to multiply together, we need to inject our current
                            // positions along the dotted axes back into the projected arg0 and arg1 coordinates.
                            std::vector<std::pair<size_t, size_t>> arg0_injection;
                            std::vector<std::pair<size_t, size_t>> arg1_injection;

                            for (size_t i = 0; i < dot_axis_pairs.size(); i++)
                            {
                                std::pair<size_t, size_t> dot_axis_pair = dot_axis_pairs[i];
                                size_t arg0_axis = dot_axis_pair.first;
                                size_t arg1_axis = dot_axis_pair.second;
                                arg0_injection.push_back(
                                    std::pair<size_t, size_t>(arg0_axis, dot_axis_positions[i]));
                                arg1_injection.push_back(
                                    std::pair<size_t, size_t>(arg1_axis, dot_axis_positions[i]));
                            }

                            Coordinate arg0_coord =
                                inject_coordinate(arg0_projected_coord, arg0_injection);
                            Coordinate arg1_coord =
                                inject_coordinate(arg1_projected_coord, arg1_injection);

                            // Multiply and add.
                            out[out_index] += arg0[arg0_transform.index(arg0_coord)] *
                                              arg1[arg1_transform.index(arg1_coord)];
                        }
                    }
                }
            }
        }
    }
}
