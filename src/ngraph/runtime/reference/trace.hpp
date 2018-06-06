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
#include <iostream>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void trace(const T* arg,
                       T* out,
                       const char* name,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape)
            {
                CoordinateTransform input_transform(arg_shape, lower_bounds, upper_bounds, strides);
                bool first = true;
                std::cout << name << " = [";
                for (const Coordinate& in_coord : input_transform)
                {
                    if (!first)
                    {
                        std::cout << ", ";
                    }
                    else
                    {
                        first = false;
                    }
                    std::cout << arg[input_transform.index(in_coord)];
                }
                std::cout << "]\n";

                //copy data
                memcpy(out, arg, sizeof(T) * shape_size(arg_shape));
            }
        }
    }
}
