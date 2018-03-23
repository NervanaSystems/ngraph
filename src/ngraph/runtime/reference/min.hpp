/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>
#include <limits>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void min(const T* arg,
                     T* out,
                     const Shape& in_shape,
                     const Shape& out_shape,
                     const AxisSet& reduction_axes)
            {
                T minval = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                                : std::numeric_limits<T>::max();

                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = minval;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = project(input_coord, reduction_axes);

                    T x = arg[input_transform.index(input_coord)];
                    T min = out[output_transform.index(output_coord)];
                    if (x < min)
                    {
                        out[output_transform.index(output_coord)] = x;
                    }
                }
            }
        }
    }
}
