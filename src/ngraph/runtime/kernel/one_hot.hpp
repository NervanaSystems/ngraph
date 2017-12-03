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
#include "ngraph/coordinate_iterator.hpp"

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
                // For the outer loop we will walk over the entire input shape.
                CoordinateIterator arg_iter(in_shape);

                do
                {
                    // For the inner loop we will walk across the entire axis for the one-hot axis, and stay put at the current arg position for the existing axes.
                    Coordinate arg_coordinate = arg_iter.get_current_coordinate();

                    Strides out_strides(out_shape.size(), 1);
                    Coordinate out_outer_corner(out_shape.size());
                    Coordinate out_inner_corner(out_shape.size());

                    size_t arg_pos = 0;

                    for (size_t i = 0; i < out_shape.size(); i++)
                    {
                        if (i != one_hot_axis)
                        {
                            // This is an existing axis.
                            out_outer_corner[i] = arg_coordinate[arg_pos];
                            out_inner_corner[i] = arg_coordinate[arg_pos];
                            arg_pos++;
                        }
                        else
                        {
                            // This is the one-hot axis.
                            out_outer_corner[i] = out_shape[i];
                            out_inner_corner[i] = 0;
                        }
                    }

                    CoordinateIterator out_iter(
                        out_shape, out_strides, out_outer_corner, out_inner_corner);

                    bool found = false;

                    do
                    {
                        auto out_index = out_iter.get_current_index();
                        auto one_hot_pos = out_iter.get_current_coordinate()[one_hot_axis];
                        auto in_index = arg_iter.get_current_index();

                        // The weird test for equality here is because this template winds up being
                        // instantiated for floating-point types, and clang complains if you try to
                        // == on a float.
                        if (arg[in_index] <= one_hot_pos && arg[in_index] >= one_hot_pos)
                        {
                            out[out_index] = 1;
                            found = true;
                        }
                        else
                        {
                            out[out_index] = 0;
                        }
                    } while (out_iter.increment());

                    if (!found)
                    {
                        throw std::range_error("One-hot: value is out of category range");
                    }
                } while (arg_iter.increment());
            }
        }
    }
}
