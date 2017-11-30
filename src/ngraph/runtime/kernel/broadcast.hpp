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
            // TODO: this function should be moved for re-use.
            Coordinate project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes)
            {
                Coordinate result;

                for (size_t i = 0; i < coord.size(); i++)
                {
                    if (deleted_axes.find(i) == deleted_axes.end())
                    {
                        result.push_back(coord[i]);
                    }
                }

                return result;
            }

            // TODO: this function should be moved for re-use.
            size_t index_in_dense_tensor(const Shape& tensor_shape, const Coordinate& coord)
            {
                size_t index = 0;
                size_t stride = 1;

                assert(tensor_shape.size() == coord.size()); // FIXME: don't use assert

                for (size_t i = tensor_shape.size(); i-- > 0;)
                {
                    assert(coord[i] <= tensor_shape[i]); // FIXME: don't use assert
                    index += coord[i] * stride;
                    stride *= tensor_shape[i];
                }

                return index;
            }

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
                    auto out_coord = out_iter.get_current_coordinate();
                    auto out_index = out_iter.get_current_index();
                    auto in_coord = project_coordinate(out_coord, broadcast_axes);
                    auto in_index = index_in_dense_tensor(in_shape, in_coord);

                    out[out_index] = arg[in_index];
                } while (out_iter.increment());
            }
        }
    }
}
