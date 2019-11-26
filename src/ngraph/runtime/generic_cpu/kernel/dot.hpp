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

#include <Eigen/Dense>
#include <cmath>
#include <utility>

#ifdef PARALLEL
#include <omp.h>
#endif

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gcpu
        {
            namespace kernel
            {
                template <typename T>
                void dot(const T* arg0,
                         const T* arg1,
                         T* out,
                         const Shape& arg0_shape,
                         const Shape& arg1_shape,
                         const Shape& out_shape,
                         size_t reduction_axes_count)
                {
                    if (arg0_shape.size() == 2 && arg1_shape.size() == 2 && out_shape.size() == 2)
                    {
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            a0(const_cast<T*>(arg0), arg0_shape[0], arg0_shape[1]);
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            a1(const_cast<T*>(arg1), arg1_shape[0], arg1_shape[1]);
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            o(const_cast<T*>(out), out_shape[0], out_shape[1]);
                        o = a0 * a1;
                    }
                    else
                    {
                        // Get the sizes of the dot axes. It's easiest to pull them from arg1
                        // because they're right up front.
                        Shape dot_axis_sizes(reduction_axes_count);
                        std::copy(arg1_shape.begin(),
                                  arg1_shape.begin() + reduction_axes_count,
                                  dot_axis_sizes.begin());

                        CoordinateTransform arg0_transform(arg0_shape);
                        CoordinateTransform arg1_transform(arg1_shape);
                        CoordinateTransform output_transform(out_shape);

                        // Create coordinate transforms for arg0 and arg1 that throw away the dotted
                        // axes.
                        size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
                        size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

                        Shape arg0_projected_shape(arg0_projected_rank);
                        std::copy(arg0_shape.begin(),
                                  arg0_shape.begin() + arg0_projected_rank,
                                  arg0_projected_shape.begin());

                        Shape arg1_projected_shape(arg1_projected_rank);
                        std::copy(arg1_shape.begin() + reduction_axes_count,
                                  arg1_shape.end(),
                                  arg1_projected_shape.begin());

                        CoordinateTransform arg0_projected_transform(arg0_projected_shape);
                        CoordinateTransform arg1_projected_transform(arg1_projected_shape);

                        // Create a coordinate transform that allows us to iterate over all possible
                        // values for the dotted axes.
                        CoordinateTransform dot_axes_transform(dot_axis_sizes);

                        for (const Coordinate& arg0_projected_coord : arg0_projected_transform)
                        {
                            for (const Coordinate& arg1_projected_coord : arg1_projected_transform)
                            {
                                // The output coordinate is just the concatenation of the projected
                                // coordinates.
                                Coordinate out_coord(arg0_projected_coord.size() +
                                                     arg1_projected_coord.size());

                                auto out_coord_it = std::copy(arg0_projected_coord.begin(),
                                                              arg0_projected_coord.end(),
                                                              out_coord.begin());
                                std::copy(arg1_projected_coord.begin(),
                                          arg1_projected_coord.end(),
                                          out_coord_it);

                                // Zero out to start the sum.
                                T sum = 0;

                                size_t out_index = output_transform.index(out_coord);

                                // Walk along the dotted axes.
                                Coordinate arg0_coord(arg0_shape.size());
                                Coordinate arg1_coord(arg1_shape.size());
                                auto arg0_it = std::copy(arg0_projected_coord.begin(),
                                                         arg0_projected_coord.end(),
                                                         arg0_coord.begin());
                                for (const Coordinate& dot_axis_positions : dot_axes_transform)
                                {
                                    // In order to find the points to multiply together, we need to
                                    // inject our current positions along the dotted axes back into
                                    // the projected arg0 and arg1 coordinates.
                                    std::copy(dot_axis_positions.begin(),
                                              dot_axis_positions.end(),
                                              arg0_it);

                                    auto arg1_it = std::copy(dot_axis_positions.begin(),
                                                             dot_axis_positions.end(),
                                                             arg1_coord.begin());
                                    std::copy(arg1_projected_coord.begin(),
                                              arg1_projected_coord.end(),
                                              arg1_it);

                                    // Multiply and add to the sum.
                                    sum += arg0[arg0_transform.index(arg0_coord)] *
                                           arg1[arg1_transform.index(arg1_coord)];
                                }

                                // Write the sum back.
                                out[out_index] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}
