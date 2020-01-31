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
                // Calculate the indices for positions 0 to rank-1.
                static void
                    get_indices(const Shape& shape, int index, std::vector<int>& indices, int rank)
                {
                    if (rank == 0)
                    {
                        return;
                    }
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

                // Gather use indices to get slices of inputs.
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
                            size_t axis,
                            int arena)
                {
                    Eigen::array<Eigen::Index, Rank1> in_dims;
                    Eigen::array<Eigen::Index, Rank2> out_dims;
                    auto axis_length = inputs_shape[axis];

                    for (size_t i = 0; i < Rank1; i++)
                    {
                        in_dims[i] = inputs_shape[i];
                    }
                    for (size_t i = 0; i < Rank2; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank2, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank1, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(inputs), in_dims);

                    auto indices_ptr = static_cast<IndicesType*>(indices);
                    auto indices_rank = indices_shape.size();
                    auto outer_loop_num = 1;
                    for (size_t i = 0; i < axis; i++)
                    {
                        outer_loop_num *= inputs_shape[i];
                    }

                    if (indices_rank == 0)
                    {
// TODO Enable this if compiler issue with CODEGEN is fixed or DEX needs it.
#if 0
#ifdef _OPENMP
#pragma omp parallel for
#endif
#endif
                        for (int i = 0; i < outer_loop_num; i++)
                        {
                            Eigen::array<Eigen::Index, Rank1> in_extents, in_offsets;
                            Eigen::array<Eigen::Index, Rank2> out_extents, out_offsets;
                            // indices_before_axis depends on inputs_shape[0,..., axis-1] and i.
                            // if axis is 0, indices_before_axis is empty.
                            std::vector<int> indices_before_axis(axis);
                            get_indices(inputs_shape, i, indices_before_axis, axis);

                            // before axis
                            for (size_t r = 0; r < axis; r++)
                            {
                                in_extents[r] = 1;
                                in_offsets[r] = indices_before_axis[r];
                            }
                            // from axis
                            for (size_t r = axis; r < Rank1; r++)
                            {
                                in_extents[r] = inputs_shape[r];
                                in_offsets[r] = 0;
                            }
                            // at axis
                            in_extents[axis] = 1;
                            // at axis, get the value from indices arg
                            IndicesType index_value = indices_ptr[0];
                            // take care of negative indices
                            in_offsets[axis] =
                                index_value >= 0 ? index_value : index_value + axis_length;

                            // before axis
                            for (size_t r = 0; r < axis; r++)
                            {
                                out_extents[r] = 1;
                                out_offsets[r] = indices_before_axis[r];
                            }
                            // after axis
                            for (size_t r = axis; r < Rank2; r++)
                            {
                                out_extents[r] = output_shape[r];
                                out_offsets[r] = 0;
                            }

                            out.slice(out_offsets, out_extents)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = in.slice(in_offsets, in_extents).reshape(out_extents);
                        }
                    }
                    else
                    {
                        size_t num_indices = 1;
                        for (size_t d : indices_shape)
                        {
                            num_indices *= d;
                        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
                        // omp requires signed iterator
                        for (int64_t i = 0; i < static_cast<int64_t>(outer_loop_num * num_indices);
                             i++)
                        {
                            Eigen::array<Eigen::Index, Rank1> in_extents, in_offsets;
                            Eigen::array<Eigen::Index, Rank2> out_extents, out_offsets;
                            std::vector<int> indices_before_axis(axis);
                            // indices_before_axis depends on inputs_shape[0,..., axis-1] and i /
                            // num_indices.
                            // if axis is 0, indices_before_axis is empty.
                            get_indices(inputs_shape, i / num_indices, indices_before_axis, axis);
                            std::vector<int> indices_from_indices_arg(indices_rank);

                            // before axis
                            for (size_t r = 0; r < axis; r++)
                            {
                                in_extents[r] = 1;
                                in_offsets[r] = indices_before_axis[r];
                            }
                            // from axis
                            for (size_t r = axis; r < Rank1; r++)
                            {
                                in_extents[r] = inputs_shape[r];
                                in_offsets[r] = 0;
                            }
                            // at axis
                            in_extents[axis] = 1;
                            // before axis
                            for (size_t r = 0; r < axis; r++)
                            {
                                out_extents[r] = 1;
                                out_offsets[r] = indices_before_axis[r];
                            }
                            // from axis
                            for (size_t r = axis; r < Rank2; r++)
                            {
                                out_extents[r] = output_shape[r];
                                out_offsets[r] = 0;
                            }
                            // at axis, get the value from indices arg
                            int k = i % num_indices;
                            IndicesType index_value = indices_ptr[k];
                            // take care of negative indices
                            in_offsets[axis] =
                                index_value >= 0 ? index_value : index_value + axis_length;

                            // indices_from_indices_arg depends on indices_shape and k.
                            // suppose the inputs has shape {3, 3, 3}, indices has shape {2, 2}, and
                            // axis is 1, the output would have shape {3, 2, 2, 3} and
                            // indices_from_indices_arg would contain indices at position 1 and 2
                            // for output slice offsets.
                            get_indices(indices_shape, k, indices_from_indices_arg, indices_rank);
                            for (size_t j = 0; j < indices_rank; j++)
                            {
                                out_extents[j + axis] = 1;
                                out_offsets[j + axis] = indices_from_indices_arg[j];
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
                                size_t axis,
                                int arena)
                {
                    gather<ElementType, int64_t, Rank1, Rank2>(inputs,
                                                               indices,
                                                               output,
                                                               inputs_shape,
                                                               indices_shape,
                                                               output_shape,
                                                               axis,
                                                               arena);
                }

                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void gather_i32(void* inputs,
                                void* indices,
                                void* output,
                                const Shape& inputs_shape,
                                const Shape& indices_shape,
                                const Shape& output_shape,
                                size_t axis,
                                int arena)
                {
                    gather<ElementType, int32_t, Rank1, Rank2>(inputs,
                                                               indices,
                                                               output,
                                                               inputs_shape,
                                                               indices_shape,
                                                               output_shape,
                                                               axis,
                                                               arena);
                }
            }
        }
    }
}
