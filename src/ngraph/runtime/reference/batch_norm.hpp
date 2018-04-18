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
#include <iostream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_norm_three_outputs(double eps,
                                          const T* arg0,
                                          const T* arg1,
                                          const T* arg2,
                                          T* out0,
                                          T* out1,
                                          T* out2,
                                          const Shape& arg2_shape)
            {
                auto eps_casted = static_cast<T>(eps);
                auto channels = arg2_shape[1];

                // We use these objects to iterate over the indices in a channel.
                // The start and end points for the channel axis are modified in the loop.
                Coordinate start_corner;
                Coordinate end_corner;
                for (size_t i = 0; i < arg2_shape.size(); i++)
                {
                    start_corner.push_back(0);
                    end_corner.push_back(arg2_shape[i]);
                }

                for (size_t c = 0; c < channels; c++)
                {
                    T channel_sum = 0;

                    start_corner[1] = c;
                    end_corner[1] = c + 1;

                    // Compute the mean
                    CoordinateTransform arg2_transform(arg2_shape, start_corner, end_corner);
                    for (Coordinate arg2_coord : arg2_transform)
                    {
                        channel_sum += arg2[arg2_transform.index(arg2_coord)];
                    }
                    T channel_mean = channel_sum / (shape_size(arg2_shape) / channels);
                    out1[c] = channel_mean;

                    // Compute the variance
                    T channel_diff_square_sum = 0;
                    for (Coordinate arg2_coord : arg2_transform)
                    {
                        auto mean_diff = arg2[arg2_transform.index(arg2_coord)] - channel_mean;
                        channel_diff_square_sum += mean_diff * mean_diff;
                    }
                    T channel_var = channel_diff_square_sum / (shape_size(arg2_shape) / channels);
                    out2[c] = channel_var;

                    // Compute the normalized output
                    for (Coordinate arg2_coord : arg2_transform)
                    {
                        auto channel_gamma = arg0[c];
                        auto channel_beta = arg1[c];

                        auto input_index = arg2_transform.index(arg2_coord);
                        auto normalized = (arg2[input_index] - channel_mean) /
                                          (std::sqrt(channel_var + eps_casted));
                        out0[input_index] = normalized * channel_gamma + channel_beta;
                    }
                }
            }

            template <typename T>
            void batch_norm_one_output(double eps,
                                       const T* arg0,
                                       const T* arg1,
                                       const T* arg2,
                                       const T* arg3,
                                       const T* arg4,
                                       T* out0,
                                       const Shape& arg2_shape)
            {
                auto eps_casted = static_cast<T>(eps);
                CoordinateTransform arg2_transform(arg2_shape);

                for (Coordinate arg2_coord : arg2_transform)
                {
                    auto channel_num = arg2_coord[1];
                    auto channel_gamma = arg0[channel_num];
                    auto channel_beta = arg1[channel_num];
                    auto channel_mean = arg3[channel_num];
                    auto channel_var = arg4[channel_num];

                    auto input_index = arg2_transform.index(arg2_coord);
                    auto normalized =
                        (arg2[input_index] - channel_mean) / (std::sqrt(channel_var + eps_casted));
                    out0[input_index] = normalized * channel_gamma + channel_beta;
                }
            }
        }
    }
}
