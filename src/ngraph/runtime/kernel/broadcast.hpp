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
            void broadcast(T* arg,
                           T* out,
                           const Shape& in_shape,
                           const Shape& out_shape,
                           const AxisSet& broadcast_axes)
            {
                // For the outer loop we will walk over the entire input shape.
                CoordinateIterator arg_iter(in_shape);

                do
                {
                    // For the inner loop we will walk across the entire axis for the new broadcast axes, and stay put at the current arg position for the existing axes.
                    Coordinate arg_coordinate = arg_iter.get_current_coordinate();

                    Strides out_strides(out_shape.size(), 1);
                    Coordinate out_outer_corner(out_shape.size());
                    Coordinate out_inner_corner(out_shape.size());

                    size_t arg_pos = 0;

                    for (size_t i = 0; i < out_shape.size(); i++)
                    {
                        if (broadcast_axes.find(i) == broadcast_axes.end())
                        {
                            // This is an existing axis.
                            out_outer_corner[i] = arg_coordinate[arg_pos];
                            out_inner_corner[i] = arg_coordinate[arg_pos];
                            arg_pos++;
                        }
                        else
                        {
                            // This is a new broadcast axis.
                            out_outer_corner[i] = out_shape[i];
                            out_inner_corner[i] = 0;
                        }
                    }

                    CoordinateIterator out_iter(
                        out_shape, out_strides, out_outer_corner, out_inner_corner);

                    do
                    {
                        out[out_iter.get_current_index()] = arg[arg_iter.get_current_index()];
                    } while (out_iter.increment());
                } while (arg_iter.increment());
            }
        }
    }
}
