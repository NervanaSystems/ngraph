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
            void sum(T* arg,
                     T* out,
                     const Shape& in_shape,
                     const Shape& out_shape,
                     const AxisSet& reduction_axes)
            {
                // Special case when the input has zero elements.
                for (size_t i = 0; i < in_shape.size(); i++)
                {
                    if (in_shape[i] == 0)
                    {
                        // Extra-special case when the output is a scalar.
                        if (out_shape.size() == 0)
                        {
                            *out = 0;
                        }
                        return;
                    }
                }

                CoordinateIterator out_iter(out_shape);

                do
                {
                    out[out_iter.get_current_index()] = 0;
                } while (out_iter.increment());

                CoordinateIterator in_iter(in_shape);

                do
                {
                    auto in_coord = in_iter.get_current_coordinate();
                    auto in_index = in_iter.get_current_index();

                    auto out_coord = project_coordinate(in_coord, reduction_axes);
                    auto out_index = index_in_dense_tensor(out_shape, out_coord);

                    out[out_index] += arg[in_index];
                } while (in_iter.increment());
            }
        }
    }
}
