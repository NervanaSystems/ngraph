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
            template <typename T, typename U>
            void non_zero(const T* arg, U* out, const Shape& arg_shape, size_t* out_count)
            {
                T zero = 0;
                size_t arg_rank = sizeof(arg_shape);
                size_t arg_count = shape_size(arg_shape);
                size_t non_zero_count = 0;

                // Count the number of non-zero items in input arg
                for (size_t i = 0; i < arg_count; i++)
                {
                    if (arg[i] > zero || arg[i] < zero)
                    {
                        non_zero_count++;
                    }
                }

                // Calculate total number of items for output
                // output is of shape {arg_rank, non_zero_count}
                *out_count = arg_rank * non_zero_count;

                if (*out_count == 0)
                {
                    // Input arg only contains 0, just return
                    return;
                }

                std::vector<size_t> elem_per_axis;
                std::vector<size_t> entry_index(arg_rank, 0);

                size_t temp = arg_count;
                size_t out_index = 0;

                for (size_t i = 0; i < arg_rank; i++)
                {
                    temp = temp / arg_shape[i];
                    elem_per_axis.push_back(temp);
                }

                // Put the non-zero item indices in out
                for (size_t i = 0; i < arg_count; i++)
                {
                    if (arg[i] > zero || arg[i] < zero)
                    {
                        temp = i;

                        for (size_t j = 0; j < arg_rank; j++)
                        {
                            temp = temp / elem_per_axis[j];
                            out_index = j * non_zero_count + entry_index[j];
                            out[out_index] = temp;

                            temp = temp % elem_per_axis[j];
                            entry_index[j]++;
                        }
                    }
                }
            }
        }
    }
}
