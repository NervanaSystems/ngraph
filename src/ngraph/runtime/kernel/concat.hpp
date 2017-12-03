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
            void concat(const std::vector<T*>& args,
                        T* out,
                        const std::vector<Shape>& in_shapes,
                        const Shape& out_shape,
                        size_t concatenation_axis)
            {
                // We will copy the inputs to the output one at a time. As we go, we will move out along the
                // concatenation axis, starting at 0.
                size_t concatenation_pos = 0;

                for (size_t i = 0; i < args.size(); i++)
                {
                    // The start coordinate for the copy is (0,...,0) except at the concatenation axis.
                    Coordinate out_start_coord = Coordinate(out_shape.size(), 0);
                    out_start_coord[concatenation_axis] = concatenation_pos;

                    // The end coordinate for the copy is the same as the output shape except at the
                    // concatenation axis.
                    Coordinate out_end_coord = out_shape;
                    out_end_coord[concatenation_axis] =
                        concatenation_pos + in_shapes[i][concatenation_axis];

                    View input_view(in_shapes[i]);
                    View output_chunk_view(out_shape, out_start_coord, out_end_coord);

                    View::Iterator output_chunk_it = output_chunk_view.begin();

                    for (Coordinate input_coord : input_view)
                    {
                        size_t input_index = input_view.index(input_coord);
                        size_t output_chunk_index = output_chunk_view.index(*output_chunk_it);
                        ++output_chunk_it;

                        out[output_chunk_index] = args[i][input_index];
                    }

                    concatenation_pos += in_shapes[i][concatenation_axis];
                }
            }
        }
    }
}
