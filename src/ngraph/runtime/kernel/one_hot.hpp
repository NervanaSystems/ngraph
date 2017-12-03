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
#include "ngraph/view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void one_hot(
                T* arg, T* out, const Shape& in_shape, const Shape& out_shape, size_t one_hot_axis)
            {
                // Step 1: Zero out the output.
                View output_view(out_shape);

                for (Coordinate output_coord : output_view)
                {
                    out[output_view.index(output_coord)] = 0;
                }

                // Step 2: Write ones at needed positions, throwing exceptions when invalid conditions
                // are encountered.
                View input_view(in_shape);

                for (Coordinate input_coord : input_view)
                {
                    T val = arg[input_view.index(input_coord)];

                    if (std::floor(val) < val || std::floor(val) > val)
                    {
                        throw(std::range_error("One-hot: non-integral value in input"));
                    }

                    size_t one_hot_pos = static_cast<size_t>(val);

                    if (one_hot_pos >= out_shape[one_hot_axis])
                    {
                        throw(std::range_error("One-hot: value is out of category range"));
                    }

                    Coordinate one_hot_coord =
                        inject_coordinate(input_coord, one_hot_axis, one_hot_pos);

                    out[output_view.index(one_hot_coord)] = 1;
                }
            }
        }
    }
}
