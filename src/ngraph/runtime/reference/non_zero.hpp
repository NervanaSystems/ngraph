//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cstddef>
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            /// \brief Return number of non-zero entries in the input argument.
            ///
            /// \param arg Input tensor
            /// \param arg_shape Input tensor shape
            /// Output number of non-zero entries in arg
            template <typename T>
            size_t non_zero_get_count(const T* arg, const Shape& arg_shape)
            {
                T zero = 0;
                size_t arg_rank = arg_shape.size();
                size_t arg_count = shape_size(arg_shape);
                size_t non_zero_count = 0;

                // Input arg is scalar
                if (arg_rank == 0)
                {
                    if (*arg != zero)
                    {
                        non_zero_count = 1;
                    }
                }
                else // Input is non scalar case
                {
                    for (size_t i = 0; i < arg_count; i++)
                    {
                        if (arg[i] != zero)
                        {
                            non_zero_count++;
                        }
                    }
                }

                return non_zero_count;
            }

            /// \brief Return indices of non-zero entries in input argument.
            ///
            /// \param arg Input tensor
            /// \param arg_shape Input tensor shape
            /// \param out Output containing indices of non-zero entries in arg
            template <typename T, typename U>
            void non_zero(const T* arg, U* out, const Shape& arg_shape)
            {
                T zero = 0;
                size_t arg_rank = arg_shape.size();
                size_t arg_count = shape_size(arg_shape);

                size_t non_zero_count = non_zero_get_count(arg, arg_shape);

                // Input arg only contains 0s
                if (non_zero_count == 0)
                {
                    return;
                }

                // Input arg is non-zero scalar
                if (arg_rank == 0)
                {
                    out[0] = static_cast<U>(0);
                    return;
                }

                // Dimensional size for the arg_shape. This is used to map one-dimentional
                // arg array indices to corresponding arg_rank-dimentional shape indices.
                // i.e., arg_shape {2, 3, 2} => elem_per_axis {6, 2, 1}.
                // Array index 4 in arg (arg[4]) correspond to 3-D index of [0][2][0]
                std::vector<size_t> elem_per_axis;
                elem_per_axis.reserve(arg_rank);

                size_t temp = arg_count;
                for (size_t i = 0; i < arg_rank; i++)
                {
                    temp = temp / arg_shape[i];
                    elem_per_axis.push_back(temp);
                }

                // Column index in out to record a non-zero entry
                size_t col_index = 0;

                // Array index in out to write non-zero index value
                size_t out_index = 0;

                // Find non-zero entries in arg and write the indices info in out.
                // For a non-zero entry, map its array index to corresponding indices
                // in arg_shape, then, write each of the arg_shape index value to
                // out array at the position that is determined by entry's dimension
                // distance and number of non-zero entries prior to it.
                // i,e., Given input with shape{2, 3, 2}, rank = 3
                // input [[[0, 2], [0, 0], [3, 4]],
                //        [[5, 0], [6, 0], [7, 8]]]
                //
                // output shape {3, 7}, rank = 2
                // output [[0, 0, 0, 1, 1, 1, 1],
                //         [0, 2, 2, 0, 1, 2, 2],
                //         [1, 0, 1, 0, 0, 0, 1]]
                //
                // input[0][2][0] = 3 is arg[4]
                // output for this entry out[1] = 0, out[8] = 2, out[15] = 0
                for (size_t i = 0; i < arg_count; i++)
                {
                    if (arg[i] != zero)
                    {
                        temp = i;

                        for (size_t j = 0; j < arg_rank; j++)
                        {
                            out_index = j * non_zero_count + col_index;
                            out[out_index] = static_cast<U>(temp / elem_per_axis[j]);

                            temp = temp % elem_per_axis[j];
                        }

                        col_index++;
                    }
                }
            }
        }
    }
}
