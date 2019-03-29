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
#include <utility>

#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_dot(const T* arg0,
                           const T* arg1,
                           T* out,
                           const Shape& arg0_shape,
                           const Shape& arg1_shape,
                           const Shape& out_shape,
                           const bool transpose0,
                           const bool transpose1)
            {
                // Create some aliases in case we need to transpose
                const T* input0 = arg0;
                const T* input1 = arg1;
                Shape input0_shape = arg0_shape;
                Shape input1_shape = arg1_shape;
                std::vector<T> input0_transposed;
                std::vector<T> input1_transposed;
                // check for transpose
                if (transpose0)
                {
                    input0_transposed.resize(shape_size(arg0_shape));
                    std::swap(input0_shape[1], input0_shape[2]);
                    reshape(
                        arg0, &input0_transposed[0], arg0_shape, AxisVector{0, 2, 1}, input0_shape);
                    input0 = &input0_transposed[0];
                }
                if (transpose1)
                {
                    input1_transposed.resize(shape_size(arg1_shape));
                    std::swap(input1_shape[1], input1_shape[2]);
                    reshape(
                        arg1, &input1_transposed[0], arg1_shape, AxisVector{0, 2, 1}, input1_shape);
                    input1 = &input1_transposed[0];
                }

                // Call dot for each pair of tensors in the batch
                const size_t batch_size = arg0_shape[0];
                const Shape dot_input0_shape{input0_shape[1], input0_shape[2]};
                const Shape dot_input1_shape{input1_shape[1], input1_shape[2]};
                const Shape dot_output_shape{out_shape[1], out_shape[2]};
                const size_t input0_offset = shape_size(dot_input0_shape);
                const size_t input1_offset = shape_size(dot_input1_shape);
                const size_t output_offset = shape_size(dot_output_shape);
                for (size_t i = 0; i < batch_size; ++i)
                {
                    dot(input0 + i * input0_offset,
                        input1 + i * input1_offset,
                        out + i * output_offset,
                        dot_input0_shape,
                        dot_input1_shape,
                        dot_output_shape,
                        1);
                }
            }
        }
    }
}
