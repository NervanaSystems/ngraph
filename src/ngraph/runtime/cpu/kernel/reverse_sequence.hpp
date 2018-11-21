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

#include <cstdint>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename InputElementType, typename SeqLenType, unsigned int Rank>
                void reverse_sequence(void* input,
                                      void* output,
                                      const Shape& input_shape,
                                      size_t batch_axis,
                                      size_t sequence_axis,
                                      void* sequence_lengths,
                                      int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }
                    Eigen::TensorMap<Eigen::Tensor<InputElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<InputElementType*>(output), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<InputElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<InputElementType*>(input), in_dims);

                    auto slv = static_cast<SeqLenType*>(sequence_lengths);

                    auto generator = [&](const Eigen::array<Eigen::DenseIndex, Rank>& i) {
                        Eigen::array<Eigen::DenseIndex, Rank> k = i;
                        if (i[sequence_axis] < slv[i[batch_axis]])
                        {
                            k[sequence_axis] = slv[i[batch_axis]] - i[sequence_axis] - 1;
                        }
                        return in(k);
                    };

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.generate(generator);
                }

                template <typename InputElementType, unsigned int Rank>
                void reverse_sequence_sli32(void* input,
                                            void* output,
                                            const Shape& input_shape,
                                            size_t batch_axis,
                                            size_t sequence_axis,
                                            void* sequence_lengths,
                                            int arena)
                {
                    reverse_sequence<InputElementType, int32_t, Rank>(input,
                                                                      output,
                                                                      input_shape,
                                                                      batch_axis,
                                                                      sequence_axis,
                                                                      sequence_lengths,
                                                                      arena);
                }
            }
        }
    }
}
