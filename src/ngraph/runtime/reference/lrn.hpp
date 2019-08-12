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
            void lrn(const T* arg,
                     const uint64_t* axes,
                     T* out,
                     const Shape& arg_shape,
                     double dalpha,
                     double dbeta,
                     double dbias,
                     size_t size)
            {
                T alpha = static_cast<T>(dalpha);
                T beta = static_cast<T>(dbeta);
                T bias = static_cast<T>(dbias);

                CoordinateTransform input_transform(arg_shape);
                const size_t current_dim = axes[0];
                const size_t elem_number_along_dim = arg_shape.at(current_dim);
                for (const Coordinate& in_coord : input_transform)
                {
                    size_t c = in_coord.at(current_dim);
                    T square_sum = 0;
                    for (size_t i = c; i < c + size; i++)
                    {
                        if (i < (size - 1) / 2)
                            continue;
                        if (i >= elem_number_along_dim + (size - 1) / 2)
                            continue;
                        auto sum_coord = in_coord;
                        sum_coord.at(current_dim) = i - (size - 1) / 2;
                        square_sum += arg[input_transform.index(sum_coord)] *
                                      arg[input_transform.index(sum_coord)];
                    }

                    T x = arg[input_transform.index(in_coord)];
                    out[input_transform.index(in_coord)] =
                        x / (std::pow(bias + (alpha / size) * square_sum, beta));
                }
            }
        }
    }
}
