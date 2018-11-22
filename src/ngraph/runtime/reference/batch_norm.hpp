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
            void batch_norm_inference(double eps,
                                      const T* gamma,
                                      const T* beta,
                                      const T* input,
                                      const T* mean,
                                      const T* variance,
                                      T* normed_input,
                                      const Shape& input_shape)
            {
                auto eps_casted = static_cast<T>(eps);
                CoordinateTransform input_transform(input_shape);

                for (Coordinate input_coord : input_transform)
                {
                    auto channel_num = input_coord[1];
                    auto channel_gamma = gamma[channel_num];
                    auto channel_beta = beta[channel_num];
                    auto channel_mean = mean[channel_num];
                    auto channel_var = variance[channel_num];

                    auto input_index = input_transform.index(input_coord);
                    auto normalized =
                        (input[input_index] - channel_mean) / (std::sqrt(channel_var + eps_casted));
                    normed_input[input_index] = normalized * channel_gamma + channel_beta;
                }
            }

            template <typename T>
            void batch_norm_training(double eps,
                                     const T* gamma,
                                     const T* beta,
                                     const T* input,
                                     T* normed_input,
                                     T* mean,
                                     T* variance,
                                     const Shape& input_shape)
            {
                std::cerr << "REFERENCE!!" << std::endl;

                auto eps_casted = static_cast<T>(eps);
                auto channels = input_shape[1];

                // We use these objects to iterate over the indices in a channel.
                // The start and end points for the channel axis are modified in the loop.
                Coordinate start_corner;
                Coordinate end_corner;
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    start_corner.push_back(0);
                    end_corner.push_back(input_shape[i]);
                }

                for (size_t c = 0; c < channels; c++)
                {
                    T channel_sum = 0;

                    start_corner[1] = c;
                    end_corner[1] = c + 1;

                    // Compute the mean
                    CoordinateTransform input_transform(input_shape, start_corner, end_corner);
                    for (Coordinate input_coord : input_transform)
                    {
                        channel_sum += input[input_transform.index(input_coord)];
                    }
                    T channel_mean = channel_sum / (shape_size(input_shape) / channels);
                    mean[c] = channel_mean;

                    T channel_diff_square_sum = 0;
                    for (Coordinate input_coord : input_transform)
                    {
                        auto centered = input[input_transform.index(input_coord)] - channel_mean;
                        channel_diff_square_sum += centered * centered;
                    }
                    T channel_var = channel_diff_square_sum / (shape_size(input_shape) / channels);
                    variance[c] = channel_var;

                    auto channel_gamma = gamma[c];
                    auto channel_beta = beta[c];
                    T scale = channel_gamma / std::sqrt(channel_var + eps_casted);

                    // Compute the normalized output
                    for (Coordinate input_coord : input_transform)
                    {
                        auto input_index = input_transform.index(input_coord);
                        normed_input[input_index] = (input[input_index] - channel_mean) * scale + channel_beta;
                    }
                }
            }

            template <typename T>
            void batch_norm_backprop(double eps,
                                     const T* gamma,
                                     const T* beta,
                                     const T* input,
                                     const T* mean,
                                     const T* variance,
                                     const T* delta_normed,
                                     T* delta_input,
                                     T* delta_gamma,
                                     T* delta_beta,
                                     const Shape& input_shape)
            {
                size_t channel_axis = 1;
                auto eps_casted = static_cast<T>(eps);
                auto num_channels = input_shape[channel_axis];
                Shape moment_shape = Shape{num_channels};
                auto input_num_elements = shape_size(input_shape);
                auto elements_per_channel = input_num_elements / num_channels;

                Coordinate start_corner;
                Coordinate end_corner;
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    start_corner.push_back(0);
                    end_corner.push_back(input_shape[i]);
                }
                // The forward computation in gory detail
                // input[., C, ...]
                // gamma[C]
                // beta[C]
                // mu[c:C] = sum(input[., c, ...])/elements_per_channel
                // centered[., c:C, ...] = input[., c, ...] - mu[c]
                // square[., c:C, ...] = centered[., c, ...]^2
                // var[c:C] = sum(centered[., c, ...]^2)/elements_per_channel
                // inv_sqrt[c:C] = 1/sqrt(var[c]+epsilon)
                // gammad[c:C] = gamma[c]*inv_sqrt[c]
                // normed[., c:C, ...] = centered[., c, ...]*gammad[c]+beta[c]

                for (auto c = 0; c < num_channels; ++c)
                {
                    start_corner[channel_axis] = c;
                    end_corner[channel_axis] = c + 1;

                    CoordinateTransform input_transform(input_shape, start_corner, end_corner);
                    std::cerr << "Elts per channel: " << elements_per_channel << std::endl;
                    T delta_beta_sum = 0;
                    T var = variance[c];
                    T mu = mean[c];
                    std::cerr << "mu: " << mu << " var: " << var << std::endl;
                    T var_eps = var+eps;
                    T sqrt_var_eps = std::sqrt(var_eps);
                    T inv_sqrt_var_eps = 1 / sqrt_var_eps;
                    T gammad = gamma[c]*inv_sqrt_var_eps;
                    T delta_gammad = 0;
                    T delta_mu = 0;
                    for (Coordinate input_coord : input_transform)
                    {
                        auto idx = input_transform.index(input_coord);
                        std::cerr << "idx: " << idx << std::endl;
                        auto delta_idx = delta_normed[idx];
                        std::cerr << "delta_idx: " << delta_idx << std::endl;
                        auto input_idx = input[idx];
                        std::cerr << "input: " << input_idx << std::endl;
                        auto centered = input_idx - mu;
                        std::cerr << "centered: " << centered << std::endl;
                        delta_beta_sum += delta_idx;
                        delta_gammad += centered * delta_idx;
                        T delta_centered = gammad*delta_idx;
                        delta_input[idx] = delta_centered;
                        delta_mu -= delta_centered;
                    }
                    delta_beta[c] = delta_beta_sum;
                    delta_gamma[c] = delta_gammad * inv_sqrt_var_eps;
                    std::cerr << "delta_gammad: " << delta_gammad << std::endl;

                    std::cerr << "di: " << delta_input[0] << " " << delta_input[1] << std::endl;
                    std::cerr << "dm: " << delta_mu << std::endl;

                    
                    T delta_inv_sqrt = gamma[c] * delta_gammad;
                    std::cerr << "delta_inv_sqrt: " << delta_inv_sqrt << std::endl;
                    T delta_var = -delta_inv_sqrt*inv_sqrt_var_eps/(2*var_eps);
                    std::cerr << "delta_var: " << delta_var << std::endl;
                    T delta_two_var_sum = 2*delta_var/elements_per_channel;
                    std::cerr << "delta_two_var_sum: " << delta_two_var_sum << std::endl;

                    for (Coordinate input_coord : input_transform)
                    {
                      auto idx = input_transform.index(input_coord);
                      auto two_centered = (input[idx]-mu)*delta_two_var_sum;
                      delta_input[idx] += two_centered;
                      delta_mu -= two_centered;
                    }
                    std::cerr << "di: " << delta_input[0] << " " << delta_input[1] << std::endl;
                    std::cerr << "dm: " << delta_mu << std::endl;


                    T delta_mu_over_n = delta_mu/elements_per_channel;
                    for (Coordinate input_coord : input_transform)
                    {
                      auto idx = input_transform.index(input_coord);
                      delta_input[idx] += delta_mu_over_n;
                    }
                    std::cerr << "di: " << delta_input[0] << " " << delta_input[1] << std::endl;
                    std::cerr << "dm: " << delta_mu << std::endl;

                }
            }
        }
    }
}
