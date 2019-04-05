//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // foreach index in params.shape[:axis]
            //    # For each slice to work on
            //    params' = param[axis:] # rank(params') == rank(params) - axis
            //    # Append dim 1 rank to indices
            //    indices' = indices.reshape(indices.shape + [1]) # indices'.shape[-1] == 1
            //    gather_nd(params', indices')
            template <typename T, typename U>
            void gather(const T* params,
                        const U* indices,
                        T* out,
                        const Shape& params_shape,
                        const Shape& indices_shape,
                        const Shape& out_shape
                        size_t axis)
            {
                using namespace std;
                // Create a CoordinateTransform for "indices" that visits only the first element along inner most axis
                size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                Coordinate indices_start_corner(indices_ndim, 0);
                Coordinate indices_inner_end_corner(indices_shape);
                size_t slice_rank = indices_shape[indices_ndim - 1];
                for(size_t i = axis; i < indices_ndim; i++)
                {
                    indices_inner_end_corner[i] = 1;
                }
                Strides indices_strides(indices_ndim, 1);
                AxisVector indices_axis_order(indices_ndim);
                iota(indices_axis_order.begin(), indices_axis_order.end(), 0);
                CoordinateTransform indices_outer_transform(
                    indices_shape, indices_start_corner, indices_inner_end_corner, indices_strides, indices_axis_order);
                Shape params_prime_shape(params_shape);
                params_prime_shape.erase(params_prime_shape.begin(), params_prime_shape.begin() + axis);
                Shape indices_prime_shape(indices_shape);
                indices_prime_shape.emplace_back(1);
                Shape out_prime_shape;
                for (const Coordinate& indices_outer_coord : indices_outer_transform)
                {
                    gather_nd(params_prime, indices_prime, out_prime, params_prime_shape, indices_prime_shape, out_prime_shape);
                }
            }
        }
    }
}
