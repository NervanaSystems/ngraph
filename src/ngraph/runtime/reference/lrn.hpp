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
#include <algorithm>

#include <iostream> //TODO

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            static void region_across_axis(
                std::vector<int64_t>& axes,
                std::vector<int>& begin_area,
                std::vector<int>& end_area,
                Coordinate& sum_coord,
                float& square_sum,
                float* arg,
                CoordinateTransform& input_transform)
            {
                if (axes.empty())
                {
                    square_sum += arg[input_transform.index(sum_coord)] *
                        arg[input_transform.index(sum_coord)];
                    return;
                }
                auto current_axis = axes.front();
                axes.erase(axes.begin());
                std::cout << "axes.size: " << axes.size() << ", current_axis: " <<current_axis << ", begin: " << begin_area[current_axis] << ", end: " << end_area[current_axis] << "\n";
                for (auto elem_index = begin_area[current_axis]; elem_index < end_area[current_axis]; ++elem_index)
                {
                    sum_coord.at(current_axis) = elem_index;
                    region_across_axis(axes, begin_area, end_area, sum_coord, square_sum, arg, input_transform);
                }
            }

            template <typename T>
            void lrn(const T* arg,
                     const AxisSet& axes,
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

                std::vector<int> begin_area(arg_shape.size());
                std::vector<int> end_area(arg_shape.size());

                CoordinateTransform input_transform(arg_shape);
                for (const Coordinate& in_coord : input_transform)
                {
                    for (const auto axis_coord : axes)
                    {
                        begin_area[axis_coord] = std::max<int>((int)0, (int)in_coord.at(axis_coord) - (int)(size - 1) / 2);
                        end_area[axis_coord] = std::min<int>((int)arg_shape.at(axis_coord), (int)in_coord.at(axis_coord) + (size - 1) / 2 + 1);
                    }
                    float square_sum = 0;
                    auto sum_coord = in_coord;
                    auto axes_vec = axes.to_vector();
                    region_across_axis(
                        axes_vec,
                        begin_area,
                        end_area,
                        sum_coord,
                        square_sum,
                        (float*)arg,
                        input_transform);

                    T x = arg[input_transform.index(in_coord)];
                    out[input_transform.index(in_coord)] =
                    x / (std::pow(bias + (alpha / size) * square_sum, beta));
                }
            }
        }
    }
}

