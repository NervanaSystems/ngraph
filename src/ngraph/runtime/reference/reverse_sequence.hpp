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
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reverse_sequence(const T* arg,
                                  T* out,
                                  const Shape& arg_shape,
                                  size_t batch_axis,
                                  size_t sequence_axis,
                                  const Shape& sequence_lengths)
            {
                CoordinateTransform input_transform(arg_shape);
                for (const Coordinate& in_coord : input_transform)
                {
                    size_t batch_index = in_coord[batch_axis];
                    //int seq_index = j < sequences[k] ? sequences[k] - j - 1 : j;
                    size_t sequence_index =
                        in_coord[sequence_axis] < sequence_lengths[batch_index]
                            ? sequence_lengths[batch_index] - in_coord[sequence_axis] - 1
                            : in_coord[sequence_axis];

                    //make a copy of in_coord and update sequence_index
                    Coordinate out_coord = in_coord;
                    out_coord[sequence_axis] = sequence_index;
                    //std::vector<size_t> tmp {out_coord.begin(), out_coord.end()};
                    std::vector<size_t> tmp {in_coord.begin(), in_coord.end()};
                    std::cout << "in coord = " << ngraph::vector_to_string(tmp)
                              <<  "batch_index = " << in_coord[batch_axis] 
                              << " sequence_lengths[batch_index] = " << sequence_lengths[batch_index]
                              << std::endl;
                    out[input_transform.index(out_coord)] = arg[input_transform.index(in_coord)];
                }
            }
        }
    }
}
