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
            void reduce(T* arg0,
                        T* arg1,
                        T* out,
                        const Shape& in_shape,
                        const Shape& out_shape,
                        const AxisSet& reduction_axes,
                        std::function<T(T, T)> reduction_function)
            {
                // General TODO: the special casing here is a bit goofy, and it's mostly a consequence of the
                // fact that a CoordinateIterator can't handle zero-length axes. (Do-while loops are a code
                // smell...) When we turn it into a proper iterator, that should go away.

                // Special case when the input has zero elements.
                for (size_t i = 0; i < in_shape.size(); i++)
                {
                    if (in_shape[i] == 0)
                    {
                        // Some input axis is zero-length; zero out the output if it is not also zero-sized.
                        for (size_t j = 0; j < out_shape.size(); j++)
                        {
                            if (out_shape[j] == 0)
                            {
                                // Some output-axis is zero length, so we don't need to do anything.
                                return;
                            }
                        }

                        // If we are still here, we need to zero out the output.
                        CoordinateIterator out_iter(out_shape);

                        do
                        {
                            out[out_iter.get_current_index()] = *arg1;
                        } while (out_iter.increment());

                        return;
                    }
                }

                CoordinateIterator out_iter(out_shape);

                do
                {
                    out[out_iter.get_current_index()] = *arg1;
                } while (out_iter.increment());

                CoordinateIterator in_iter(in_shape);

                do
                {
                    Coordinate in_coord = in_iter.get_current_coordinate();
                    size_t in_index = in_iter.get_current_index();

                    Coordinate out_coord = project_coordinate(in_coord, reduction_axes);
                    size_t out_index = index_in_dense_tensor(out_shape, out_coord);

                    out[out_index] = reduction_function(out[out_index], arg0[in_index]);
                } while (in_iter.increment());
            }
        }
    }
}
