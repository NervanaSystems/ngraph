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
            void dot(T* arg0,
                     T* arg1,
                     T* out,
                     const Shape& arg0_shape,
                     const Shape& arg1_shape,
                     const Shape& out_shape,
                     size_t arg0_dot_axis,
                     size_t arg1_dot_axis)
            {
                CoordinateTransform output_transform(out_shape);

                for (Coordinate out_coord : output_transform)
                {
                    out[output_transform.index(out_coord)] = 0;
                }

                CoordinateTransform arg0_transform(arg0_shape);
                CoordinateTransform arg1_transform(arg1_shape);

                CoordinateTransform arg0_projected_transform(
                    project_shape(arg0_shape, AxisSet{arg0_dot_axis}));
                CoordinateTransform arg1_projected_transform(
                    project_shape(arg1_shape, AxisSet{arg1_dot_axis}));

                for (Coordinate arg0_projected_coord : arg0_projected_transform)
                {
                    for (Coordinate arg1_projected_coord : arg1_projected_transform)
                    {
                        for (size_t i = 0; i < arg0_shape[arg0_dot_axis]; i++)
                        {
                            Coordinate arg0_coord =
                                inject_coordinate(arg0_projected_coord, arg0_dot_axis, i);
                            Coordinate arg1_coord =
                                inject_coordinate(arg1_projected_coord, arg1_dot_axis, i);

                            Coordinate out_coord(arg0_projected_coord.size() +
                                                 arg1_projected_coord.size());

                            std::copy(arg0_projected_coord.begin(),
                                      arg0_projected_coord.end(),
                                      out_coord.begin());
                            std::copy(arg1_projected_coord.begin(),
                                      arg1_projected_coord.end(),
                                      out_coord.begin() + arg0_projected_coord.size());

                            out[output_transform.index(out_coord)] +=
                                arg0[arg0_transform.index(arg0_coord)] *
                                arg1[arg1_transform.index(arg1_coord)];
                        }
                    }
                }
            }
        }
    }
}
