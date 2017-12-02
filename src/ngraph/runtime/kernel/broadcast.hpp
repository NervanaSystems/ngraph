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
                CoordinateIterator out_iter(out_shape);

                do
                {
                    Coordinate out_coord = out_iter.get_current_coordinate();
                    size_t out_index = out_iter.get_current_index();
                    Coordinate in_coord = project_coordinate(out_coord, broadcast_axes);
                    size_t in_index = index_in_dense_tensor(in_shape, in_coord);

                    out[out_index] = arg[in_index];
                } while (out_iter.increment());
            }
        }
    }
}
