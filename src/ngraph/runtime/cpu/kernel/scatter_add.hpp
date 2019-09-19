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
#include "ngraph/runtime/reference/scatter_add.hpp"
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
                    for (size_t j = 0; j < rank; j++)
                    {
                        indices[j] = index / partial_sum[j];
                        index = index % partial_sum[j];
                    }
                }

                // ScatterAdd is to update bunch of slices of the inputs. The rank of slice is 1
                // less than the rank of the inputs.
                template <typename ElementType,
                          typename IndicesType,
                          unsigned int Rank1,
                          unsigned int Rank2>
                void scatter_add(void* inputs,
                                 void* indices,
                                 void* updates,
                                 void* output,
                                 const Shape& inputs_shape,
                                 const Shape& indices_shape,
                                 const Shape& updates_shape,
                                 int arena)
                {
                    // For Eigen slice op, both parameters (offsets and extents) need to have the
                    // same rank.
                    // Here *_offsets and *_extents have the same rank.
                    Eigen::array<Eigen::Index, Rank1> in_dims, in_extents, in_offsets;
                    Eigen::array<Eigen::Index, Rank2> updates_dims, updates_extents,
                        updates_offsets;

                    for (size_t i = 0; i < Rank1; i++)
                    {
                        in_extents[i] = in_dims[i] = inputs_shape[i];
                        in_offsets[i] = 0;
                    }
                    in_extents[0] = 1;
                    for (size_t i = 0; i < Rank2; i++)
                    {
                        updates_extents[i] = updates_dims[i] = updates_shape[i];
                        updates_offsets[i] = 0;
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank1, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(inputs), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank2, Eigen::RowMajor>> up(
                        static_cast<ElementType*>(updates), updates_dims);

                    // copy if not in place.
                    if (inputs != output)
                    {
                        out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = in;
                    }

                    auto indices_ptr = static_cast<IndicesType*>(indices);
                    auto indices_rank = indices_shape.size();
                    if (indices_rank == 0)
                    {
                        in_offsets[0] = indices_ptr[0];
                        out.slice(in_offsets, in_extents)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                arena)) =
                            out.slice(in_offsets, in_extents) +
                            up.slice(updates_offsets, updates_extents).reshape(in_extents);
                    }
                    else
                    {
                        std::vector<int> leading_indices(indices_rank);
                        for (size_t i = 0; i < shape_size(indices_shape); i++)
                        {
                            in_offsets[0] = indices_ptr[i];
                            get_leading_indices(indices_shape, i, leading_indices);
                            for (size_t j = 0; j < indices_rank; j++)
                            {
                                updates_extents[j] = 1;
                                updates_offsets[j] = leading_indices[j];
                            }
                            out.slice(in_offsets, in_extents)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) =
                                out.slice(in_offsets, in_extents) +
                                up.slice(updates_offsets, updates_extents).reshape(in_extents);
                        }
                    }
                }

                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void scatter_add_i64(void* inputs,
                                     void* indices,
                                     void* updates,
                                     void* output,
                                     const Shape& inputs_shape,
                                     const Shape& indices_shape,
                                     const Shape& updates_shape,
                                     int arena)
                {
                    scatter_add<ElementType, int64_t, Rank1, Rank2>(inputs,
                                                                    indices,
                                                                    updates,
                                                                    output,
                                                                    inputs_shape,
                                                                    indices_shape,
                                                                    updates_shape,
                                                                    arena);
                }

                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void scatter_add_i32(void* inputs,
                                     void* indices,
                                     void* updates,
                                     void* output,
                                     const Shape& inputs_shape,
                                     const Shape& indices_shape,
                                     const Shape& updates_shape,
                                     int arena)
                {
                    scatter_add<ElementType, int32_t, Rank1, Rank2>(inputs,
                                                                    indices,
                                                                    updates,
                                                                    output,
                                                                    inputs_shape,
                                                                    indices_shape,
                                                                    updates_shape,
                                                                    arena);
                }

                template <typename ElementType>
                void ref_scatter_add_i32(void* inputs,
                                         void* indices,
                                         void* updates,
                                         void* output,
                                         const Shape& inputs_shape,
                                         const Shape& indices_shape,
                                         const Shape& updates_shape,
                                         const Shape& output_shape)
                {
                    reference::scatter_add<ElementType, int32_t>(static_cast<ElementType*>(inputs),
                                                                 static_cast<int32_t*>(indices),
                                                                 static_cast<ElementType*>(updates),
                                                                 static_cast<ElementType*>(output),
                                                                 inputs_shape,
                                                                 indices_shape,
                                                                 updates_shape,
                                                                 output_shape);
                }

                template <typename ElementType>
                void ref_scatter_add_i64(void* inputs,
                                         void* indices,
                                         void* updates,
                                         void* output,
                                         const Shape& inputs_shape,
                                         const Shape& indices_shape,
                                         const Shape& updates_shape,
                                         const Shape& output_shape)
                {
                    reference::scatter_add<ElementType, int64_t>(static_cast<ElementType*>(inputs),
                                                                 static_cast<int64_t*>(indices),
                                                                 static_cast<ElementType*>(updates),
                                                                 static_cast<ElementType*>(output),
                                                                 inputs_shape,
                                                                 indices_shape,
                                                                 updates_shape,
                                                                 output_shape);
                }
            }
        }
    }
}
