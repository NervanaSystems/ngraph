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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // NOTE: Execution throws `std::range_error` if either a non-integral value or an out-of-bounds
            // value is detected in the input tensor.
            template <typename T>
            void one_hot(const T* arg,
                         T* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         size_t one_hot_axis)
            {
                // Step 1: Zero out the output.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 0;
                }

                // Step 2: Write ones at needed positions, throwing exceptions when invalid conditions
                // are encountered.
                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    T val = arg[input_transform.index(input_coord)];

                    if (std::floor(val) < val || std::floor(val) > val)
                    {
                        throw(std::range_error("One-hot: non-integral value in input"));
                    }

                    size_t one_hot_pos = static_cast<size_t>(val);

                    if (one_hot_pos >= out_shape[one_hot_axis])
                    {
                        throw(std::range_error("One-hot: value is out of category range"));
                    }

                    Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);

                    out[output_transform.index(one_hot_coord)] = 1;
                }
            }
        }
    }
}
