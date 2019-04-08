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
            // Implement gather by calling gather_nd on sub-problems
            // # prepare constant shapes for tensors used for sub problems
            // indices'.shape  = indices.shape[-1] + [1]
            // params'.shape = params.shape[axis:]
            // out'.shape = params'.shape
            // out'.shape[0] = indices.shape[-1]
            // # call sub-problems
            // foreach (params_index, out_index) in outer "axis" dimensions
            //     # params_prime is shared by inner loop
            //     params' = param[params_index] # rank(params') == rank(params) - axis
            //     foreach indices_index in outer N-1 dimensions
            //         indices' = indices[indices_index] # rank(indices') == 2
            //         out_index = out_index + indices_index
            //         out' = out[out_index] # rank(out') == rank(params')
            //         gather_nd(params', indices'', out')
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
                // prepare shape of indices_prime (2-D)
                size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                Shape indices_prime_shape({indices_shape[indices_ndim - 1], 1});
                // prepare shape of params_prime (remove first "axis" dimensions)
                Shape params_prime_shape(params_shape);
                params_prime_shape.erase(params_prime_shape.begin(), params_prime_shape.begin() + axis);
                // prepare shape of out_prime (same as params_prime except for first dim)
                Shape out_prime_shape(params_prime_shape);
                out_prime_shape[0] = indices_shape[indices_ndim - 1];


                // Create a CoordinateTransform for "indices" that visits only the first element along inner most axis
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

                // Create a matching CoordinateTransform for "out" that visits the same outer coordinates
                size_t out_ndim = static_cast<size_t>(out_shape.size());
                Coordinate out_start_corner(out_ndim, 0);
                Coordinate out_end_corner(out_shape);
                for(size_t i = axis - 1; i < out_ndim; i++)
                {
                    out_inner_end_corner[i] = 1;
                }
                Strides out_strides(out_ndim, 1);
                AxisVector out_axis_order(out_ndim);
                iota(out_axis_order.begin(), out_axis_order.end(), 0);
                CoordinateTransform out_transform(
                    out_shape, out_start_corner, out_inner_end_corner, out_strides, out_axis_order);

                for (const Coordinate& indices_outer_coord : indices_outer_transform)
                {
                    indices_prime = &indices[indices_outer_transform.index(indices_outer_coord)];
                    gather_nd(params_prime, indices_prime, out_prime, params_prime_shape, indices_prime_shape, out_prime_shape);
                }
                // reshape out
                //
            }
        }
    }
}
