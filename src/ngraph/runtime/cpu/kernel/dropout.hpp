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

#include <random>

#include "ngraph/state/rng_state.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                // Note: this kernel is for doing upscale in train
                /*template <typename T>
                void generate_dropout(T* input,
                                    T* out0,
                                    T* out1_mask,
                                    size_t count,
                                    const Shape& input_shape, // input shape is for future opt
                                    ngraph::RNGState* rng_state,
                                    bool training,
                                    const double value)
                {
                    auto& gen = rng_state->get_generator();
                    auto& bd = rng_state->get_distribution();

                    if (training)
                    {
                        double dropout_prob = 1 - value;
                        for (size_t i = 0; i < count; ++i)
                        {
                            if (static_cast<T>(bd(gen)) < dropout_prob) {
                                out1_mask[i] = 0;
                                out0[i] = 0;
                            } else {
                                out1_mask[i] = 1; 
                                out0[i] = input[i] / static_cast<T>(value);
                            }
                        }
                    }
                    else
                    {
                        // this is inference, so, no need for mask output is not needed
                        for (size_t i = 0; i < count; i++)
                        {
                            out0[i] = static_cast<T>(1);
                        }
                    }
                }*/

                template <typename T>
                void generate_dropout(T* input,
                                    T* out0,
                                    T* out1_mask,
                                    size_t count,
                                    const Shape& input_shape, // input shape is for future opt
                                    ngraph::RNGUniformState* rng_state,
                                    bool training,
                                    const double value)
                {
                    auto& gen = rng_state->get_generator();
                    auto& bd = rng_state->get_distribution();

                    if (training)
                    {
                        double dropout_prob = 1 - value;
                        for (size_t i = 0; i < count; ++i)
                        {
                            if (static_cast<T>(bd(gen)) < dropout_prob) {
                                out1_mask[i] = 0;
                                out0[i] = 0;
                            } else {
                                out1_mask[i] = 1; 
                                out0[i] = input[i] / static_cast<T>(value);
                            }
                        }
                    }
                    else
                    {
                        // this is inference, so, no need for mask output is not needed
                        for (size_t i = 0; i < count; i++)
                        {
                            out0[i] = static_cast<T>(1);
                        }
                    }
                }

            }
        }
    }
}
