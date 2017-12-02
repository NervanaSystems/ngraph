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
            void concat(const std::vector<T*>& args,
                        T* out,
                        const std::vector<Shape>& in_shapes,
                        const Shape& out_shape,
                        size_t concatenation_axis)
            {
                size_t concatenation_pos = 0;

                for (size_t i = 0; i < args.size(); i++)
                {
                    Strides strides(out_shape.size(), 1);

                    Coordinate out_start_coord = Coordinate(out_shape.size(), 0);
                    out_start_coord[concatenation_axis] = concatenation_pos;
                    Coordinate out_end_coord = out_shape;
                    out_end_coord[concatenation_axis] =
                        concatenation_pos + in_shapes[i][concatenation_axis];

                    CoordinateIterator out_iter(out_shape, strides, out_end_coord, out_start_coord);
                    CoordinateIterator in_iter(in_shapes[i]);

                    do
                    {
                        out[out_iter.get_current_index()] = args[i][in_iter.get_current_index()];
                    } while (out_iter.increment() && in_iter.increment());

                    concatenation_pos += in_shapes[i][concatenation_axis];
                }
            }
        }
    }
}
