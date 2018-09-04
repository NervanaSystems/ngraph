/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>
#include <algorithm>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void topk(
                const T* arg, U* out_indices, T* out_values, const Shape& in_shape, const Shape& out_shape, size_t axis, size_t k, bool compute_max)
            {
                // reorder source axis visit order and make "axis" inner most
                size_t ndim = static_cast<size_t>(in_shape.size());
                Coordinate start_corner(ndim, 0);
                Coordinate end_corner(in_shape);
                end_corner[axis] = 1;
                Strides strides(ndim, 1);
                AxisVector axis_order(ndim);
                std::iota(axis_order.begin(), axis_order.end(), 0);
                axis_order.erase(axis_order.begin() + axis);
                axis_order.push_back(axis);
                // Create CoordinateTransforms that visits only the first element along "axis"
                CoordinateTransform input_transform(in_shape,
                        start_corner,
                        end_corner,
                        strides,
                        axis_order);
                CoordinateTransform output_transform(out_shape,
                        start_corner,
                        end_corner,
                        strides,
                        axis_order);
                // Create temp vector for sorting.
                std::vector<std::tuple<T, U>> workspace(in_shape[axis]);
                std::vector<size_t> in_strides = ngraph::row_major_strides(in_shape);
                std::vector<size_t> out_strides = ngraph::row_major_strides(out_shape);
                auto in_axis_stride = in_strides[axis];
                auto out_axis_stride = out_strides[axis];
                for(const Coordinate& coord: input_transform)
                {
                    auto arg_index = input_transform.index(coord);
                    auto out_index = output_transform.index(coord);
                    // Fill the temp vector
                    U i = 0;
                    for(std::tuple<T, U> &entry : workspace)
                    {
                        // std::cout << "Read arg[" << arg_index << "]" <<std::endl;
                        std::get<0>(entry) = arg[arg_index];
                        std::get<1>(entry) = i;
                        arg_index += in_axis_stride;
                        i++;
                    }
                    // Sort the temp vector
                    std::sort(workspace.begin(),
                            workspace.end(),
                            compute_max ?
                                [] (const std::tuple<T, U>&  a, const std::tuple<T, U>& b) -> bool { return a > b;} :
                                [] (const std::tuple<T, U>&  a, const std::tuple<T, U>& b) -> bool { return a < b;});
                    // Write temp vector to output
                    for(size_t i = 0; i < k ; i++)
                    {
                        std::tuple<T, U> entry = workspace[i];
                        // std::cout << "Write out[" << out_index << "]" <<std::endl;
                        out_values[out_index]=std::get<0>(entry);
                        out_indices[out_index]=std::get<1>(entry);
                        out_index += out_axis_stride;
                    }
                }
            }
        }
    }
}
