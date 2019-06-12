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

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/coordinate.hpp"
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
                static void
                    get_leading_indices(const Shape& shape, int index, std::vector<int>& indices)
                {
                    auto rank = shape.size();
                    std::vector<int> partial_sum(rank);

                    partial_sum[rank - 1] = 1;
                    for (int j = rank - 2; j >= 0; j--)
                    {
                        partial_sum[j] = partial_sum[j + 1] * shape[j + 1];
                    }
                    for (int j = 0; j < rank; j++)
                    {
                        indices[j] = index / partial_sum[j];
                        index = index % partial_sum[j];
                    }
                }

                template <typename ElementType,
                          typename IndicesType,
                          unsigned int Rank1,
                          unsigned int Rank2>
                void gather(void* inputs,
                            void* indices,
                            void* output,
                            const Shape& inputs_shape,
                            const Shape& indices_shape,
                            const Shape& output_shape,
                            int arena)
                {
                    Eigen::array<Eigen::Index, Rank1> in_dims;
                    Eigen::array<Eigen::Index, Rank2> out_dims;

                    for (int i = 0; i < Rank1; i++)
                    {
                        in_dims[i] = inputs_shape[i];
                    }
                    for (int i = 0; i < Rank2; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank2, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank1, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(inputs), in_dims);

                    auto indices_ptr = static_cast<IndicesType*>(indices);
                    auto indices_rank = indices_shape.size();

                    if (indices_rank == 0)
                    {
                        Eigen::array<Eigen::Index, Rank1> in_extents, in_offsets;
                        Eigen::array<Eigen::Index, Rank2> out_extents, out_offsets;

                        for (int i = 0; i < Rank1; i++)
                        {
                            in_extents[i] = inputs_shape[i];
                            in_offsets[i] = 0;
                        }
                        in_extents[0] = 1;
                        in_offsets[0] = indices_ptr[0];
                        for (int i = 0; i < Rank2; i++)
                        {
                            out_extents[i] = output_shape[i];
                            out_offsets[i] = 0;
                        }

                        out.slice(out_offsets, out_extents)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                arena)) = in.slice(in_offsets, in_extents).reshape(out_extents);
                    }
                    else
                    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                        for (int i = 0; i < shape_size(indices_shape); i++)
                        {
                            // Declare these inside the loop for omp parallel
                            Eigen::array<Eigen::Index, Rank1> in_extents, in_offsets;
                            Eigen::array<Eigen::Index, Rank2> out_extents, out_offsets;
                            std::vector<int> leading_indices(indices_rank);

                            for (int r = 0; r < Rank1; r++)
                            {
                                in_extents[r] = inputs_shape[r];
                                in_offsets[r] = 0;
                            }
                            in_extents[0] = 1;
                            in_offsets[0] = indices_ptr[i];

                            for (int r = 0; r < Rank2; r++)
                            {
                                out_extents[r] = output_shape[r];
                                out_offsets[r] = 0;
                            }
                            get_leading_indices(indices_shape, i, leading_indices);
                            for (int j = 0; j < indices_rank; j++)
                            {
                                out_extents[j] = 1;
                                out_offsets[j] = leading_indices[j];
                            }
                            out.slice(out_offsets, out_extents)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = in.slice(in_offsets, in_extents).reshape(out_extents);
                        }
                    }
                }

                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void gather_i64(void* inputs,
                                void* indices,
                                void* output,
                                const Shape& inputs_shape,
                                const Shape& indices_shape,
                                const Shape& output_shape,
                                int arena)
                {
                    gather<ElementType, int64_t, Rank1, Rank2>(
                        inputs, indices, output, inputs_shape, indices_shape, output_shape, arena);
                }

                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void gather_i32(void* inputs,
                                void* indices,
                                void* output,
                                const Shape& inputs_shape,
                                const Shape& indices_shape,
                                const Shape& output_shape,
                                int arena)
                {
                    gather<ElementType, int32_t, Rank1, Rank2>(
                        inputs, indices, output, inputs_shape, indices_shape, output_shape, arena);
                }
            }
        }
    }
}
