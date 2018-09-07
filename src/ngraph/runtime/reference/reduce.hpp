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
            void reduce(const T* arg0,
                        const T* arg1, // TODO: really we should just pass a T here.
                        T* out,
                        const Shape& in_shape,
                        const Shape& out_shape,
                        const AxisSet& reduction_axes,
                        std::function<T(T, T)> reduction_function)
            {
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = *arg1;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = project(input_coord, reduction_axes);
                    size_t input_index = input_transform.index(input_coord);
                    size_t output_index = output_transform.index(output_coord);

                    out[output_index] = reduction_function(out[output_index], arg0[input_index]);
                }
            }
        }
    }
}
