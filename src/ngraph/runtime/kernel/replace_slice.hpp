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
                CoordinateIterator copy_iter(out_shape);

                do
                {
                    out[copy_iter.get_current_index()] = arg0[copy_iter.get_current_index()];
                } while (copy_iter.increment());

                // Step 2: Overwrite the slice for replacement.
                CoordinateIterator out_iter(out_shape, strides, upper_bounds, lower_bounds);
                CoordinateIterator in_iter(arg1_shape);

                do
                {
                    auto out_index = out_iter.get_current_index();
                    auto in_index = in_iter.get_current_index();

                    out[out_index] = arg1[in_index];
                } while (out_iter.increment() && in_iter.increment());
            }
        }
    }
}
