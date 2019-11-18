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

#include <cmath>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void cumsum(const T* arg,
                        T* out,
                        const Shape& in_shape,
                        const Shape& out_shape,
                        const int64_t axis,
                        const int exclusive,
                        const int reverse)
            {
                CoordinateTransform temp_transform(out_shape);
                for (const Coordinate& output_coord : temp_transform)
                {
                    out[temp_transform.index(output_coord)] = 0;
                }

                auto get_key = [&, axis](const Coordinate& coord) -> Coordinate {
                    Coordinate result(coord.size(), 0);
                    result[axis] = coord[axis];

                    for (size_t i = 0; i < coord.size(); i++)
                    {
                        result[i] = coord[i] - result[i];
                    }
                    return result;
                };

                auto cum_sum =
                    [&, exclusive, reverse](std::vector<std::pair<size_t, T>>& tensor_vec) {
                        // TODO (pthoreho): Add support for exclsuive and reverse mode
                        if (reverse == 0)
                        {
                            T prev = 0;
                            for (size_t i = 0; i < tensor_vec.size(); i++)
                            {
                                tensor_vec[i].second = prev + tensor_vec[i].second;
                                out[tensor_vec[i].first] = tensor_vec[i].second;
                                prev = out[tensor_vec[i].first];
                            }
                        }
                    };

                // Map to collect tensor elements belonging to the same axis
                std::map<Coordinate, std::vector<std::pair<size_t, T>>> map_cooord_to_val;
                CoordinateTransform input_transform(in_shape);
                for (const Coordinate& input_coord : input_transform)
                {
                    // points to the current element in the input tensor
                    T current = arg[input_transform.index(input_coord)];
                    auto key = get_key(input_coord);
                    auto index = input_transform.index(input_coord);
                    if (map_cooord_to_val.find(key) != map_cooord_to_val.end())
                    {
                        map_cooord_to_val[key].push_back(std::make_pair(index, current));
                    }
                    else
                    {
                        map_cooord_to_val.insert({key, std::vector<std::pair<size_t, T>>()});
                        map_cooord_to_val[key].push_back(std::make_pair(index, current));
                    }
                }
                // iterate the map and perform cumulative sum over the give axis
                for (auto& it : map_cooord_to_val)
                {
                    cum_sum(it.second);
                }
            }
        }
    }
}
