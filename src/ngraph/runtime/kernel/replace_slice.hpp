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
            void replace_slice(T* arg0, // replacement context
                               T* arg1, // replacement value
                               T* out,
                               const Shape& arg1_shape,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Strides& strides,
                               const Shape& out_shape)
            {
                // Step 1: Copy the entire replacement context to the output.
                View copy_view(out_shape);

                for (Coordinate copy_coord : copy_view)
                {
                    out[copy_view.index(copy_coord)] = arg0[copy_view.index(copy_coord)];
                }

                // Step 2: Overwrite the slice for replacement.
                View input_view(arg1_shape);
                View output_view(out_shape, lower_bounds, upper_bounds, strides);

                View::Iterator output_it = output_view.begin();

                for (Coordinate input_coord : input_view)
                {
                    Coordinate output_coord = *output_it;
                    ++output_it;

                    out[output_view.index(output_coord)] = arg1[input_view.index(input_coord)];
                }
            }
        }
    }
}
