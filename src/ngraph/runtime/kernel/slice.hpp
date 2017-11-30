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
            void slice(T* arg,
                       T* out,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape)
            {
                CoordinateIterator in_iter(arg_shape, strides, upper_bounds, lower_bounds);
                CoordinateIterator out_iter(out_shape);

                do
                {
                    auto out_index = out_iter.get_current_index();
                    auto in_index = in_iter.get_current_index();

                    out[out_index] = arg[in_index];
                } while (out_iter.increment() && in_iter.increment());
            }
        }
    }
}
