//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include <iostream>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_norm_three_outputs_with_intermediates(double eps,
                                                             const T* arg0,
                                                             const T* arg1,
                                                             const T* arg2,
                                                             T* out0,
                                                             T* out1,
                                                             T* out2,
                                                             T* out3,
                                                             T* out4,
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
                        out3[input_index] = arg2[input_index] - channel_mean;
                        out4[input_index] =
                            out3[input_index] / (std::sqrt(channel_var + eps_casted));
                        out0[input_index] = out4[input_index] * channel_gamma + channel_beta;
                    }
                }
            }

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
                std::vector<T> centered(shape_size(arg2_shape));
                std::vector<T> normalized(shape_size(arg2_shape));
                batch_norm_three_outputs_with_intermediates(eps,
                                                            arg0,
                                                            arg1,
                                                            arg2,
                                                            out0,
                                                            out1,
                                                            out2,
                                                            centered.data(),
                                                            normalized.data(),
                                                            arg2_shape);
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

            template <typename T>
            void batch_norm_backprop(double eps,
                                     const T* arg0,
                                     const T* arg1,
                                     const T* arg2,
                                     const T* arg3,
                                     const T* arg4,
                                     const T* arg5,
                                     T* out0,
                                     T* out1,
                                     T* out2,
                                     const Shape& arg2_shape)
            {
                auto eps_casted = static_cast<T>(eps);

                Shape mean_shape{arg2_shape[1]};
                AxisSet reduction_axes;
                for (size_t idx = 0; idx < arg2_shape.size(); idx++)
                {
                    if (idx != 1)
                    {
                        reduction_axes.insert(idx);
                    }
                }
                auto arg2_num_elements = shape_size(arg2_shape);
                auto mean_num_elements = shape_size(mean_shape);
                auto reduction_axes_size = arg2_num_elements / mean_num_elements;

                // Compute the mean, variance, and normalized values

                std::vector<T> bn_output(arg2_num_elements);
                std::vector<T> centered(arg2_num_elements);
                std::vector<T> normalized(arg2_num_elements);

                std::vector<T> mean(mean_num_elements);
                std::vector<T> variance(mean_num_elements);
                std::vector<T> stddev(mean_num_elements);
                batch_norm_three_outputs_with_intermediates(eps,
                                                            arg0,
                                                            arg1,
                                                            arg2,
                                                            bn_output.data(),
                                                            mean.data(),
                                                            variance.data(),
                                                            centered.data(),
                                                            normalized.data(),
                                                            arg2_shape);

                for (size_t i = 0; i < mean_num_elements; i++)
                {
                    stddev[i] = std::sqrt(variance[i] + eps_casted);
                }

                // Broadcast gamma and the standard deviation
                std::vector<T> gamma_bcast(arg2_num_elements);
                std::vector<T> stddev_bcast(arg2_num_elements);
                broadcast(arg0, gamma_bcast.data(), mean_shape, arg2_shape, reduction_axes);
                broadcast(
                    stddev.data(), stddev_bcast.data(), mean_shape, arg2_shape, reduction_axes);

                // Bprop into gamma
                std::vector<T> delta_times_normalized(arg2_num_elements);
                multiply(normalized.data(), arg5, delta_times_normalized.data(), arg2_num_elements);
                sum(delta_times_normalized.data(), out1, arg2_shape, mean_shape, reduction_axes);

                // Bprop into beta
                sum(arg5, out2, arg2_shape, mean_shape, reduction_axes);

                // // Broadcast the gamma and beta grads
                std::vector<T> delta_gamma_bcast(arg2_num_elements);
                broadcast(out1, delta_gamma_bcast.data(), mean_shape, arg2_shape, reduction_axes);
                std::vector<T> delta_beta_bcast(arg2_num_elements);
                broadcast(out2, delta_beta_bcast.data(), mean_shape, arg2_shape, reduction_axes);

                // Bprop into the input
                for (size_t i = 0; i < arg2_num_elements; i++)
                {
                    auto scale_normalized = gamma_bcast[i] / stddev_bcast[i];
                    out0[i] = static_cast<T>(
                        scale_normalized *
                        (arg5[i] -
                         (normalized[i] * delta_gamma_bcast[i] + delta_beta_bcast[i]) /
                             reduction_axes_size));
                }
            }
        }
    }
}
