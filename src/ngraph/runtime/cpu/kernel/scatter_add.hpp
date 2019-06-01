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
                // ScatterAdd is to update bunch of slices of the inputs. The rank of slice is 1 less than the rank of the inputs.

                // For Eigen slice op, both parameters (offsets and extents) need to have the same rank.
                // Here *_indices and *_slice_dims have the same rank.
                // Suppose the slice would have shape {3, 3} ignoring the leading 1s.
                // And if the shape of *_indices has zeros at position 0, ..., rank - 1 - 2(rank of slice without leading 1s),
                // the shape of *_slice_dimes should have ones at the same position.
                // These works: {1, 1, 0, 0} and {1, 1, 3, 3}
                //              {0, 0, 0, 0} and {1, 1, 3, 3}
                //              {0, 0, 0} and {1, 3, 3}
                //              {0, 0} and {3, 3}
                // This does not work: {0, 0, 0, 0} and {1, 0, 3, 3}
                // Change 0 to 1 at those positions:
                // For in_slice_dims, we only need to check the its shape[0] since its rank is always the rank of slice without leading 1s + 1.
                // For updates_slice_dims, we need to check its shape[0], ..., shape[rank of indices - 1] if indices is not scalar.

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
                    Eigen::array<Eigen::Index, Rank1> in_dims, in_slice_dims, in_indices;
                    Eigen::array<Eigen::Index, Rank2> updates_dims, updates_slice_dims,
                        updates_indices;

                    for (int i = 0; i < Rank1; i++)
                    {
                        in_slice_dims[i] = in_dims[i] = inputs_shape[i];
                        in_indices[i] = 0;
                    }
                    for (int i = 0; i < Rank2; i++)
                    {
                        updates_slice_dims[i] = updates_dims[i] = updates_shape[i];
                        updates_indices[i] = 0;
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
                    auto size = indices_shape.size();
                    if (size == 0)
                    {
                        in_slice_dims[0] = in_indices[0] = indices_ptr[0];
                        // change to 1 if 0.
                        in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                        out.slice(in_indices, in_slice_dims)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                arena)) = out.slice(in_indices, in_slice_dims) +
                                          up.slice(updates_indices, updates_slice_dims);
                    }
                    else
                    {
                        // This is used to calculate the leading indices for updates_indices and updates_slice_dims
                        // when size > 1.
                        std::vector<int> ele_counts(size);
                        ele_counts[size - 1] = 0;
                        if (size > 1)
                        {
                            ele_counts[size - 2] = indices_shape[size - 1];
                        }
                        for (int j = size - 3; j >= 0; j--)
                        {
                            ele_counts[j] = indices_shape[j + 2] * indices_shape[j + 1];
                        }
                        for (int i = 0; i < shape_size(indices_shape); i++)
                        {
                            in_slice_dims[0] = in_indices[0] = indices_ptr[i];
                            int k = i;
                            for (int j = 0; j < size - 1; j++)
                            {
                                updates_slice_dims[j] = updates_indices[j] = k / ele_counts[j];
                                // change to 1 if 0.
                                updates_slice_dims[j] =
                                    updates_slice_dims[j] == 0 ? 1 : updates_slice_dims[j];
                                k = k % ele_counts[j];
                            }
                            updates_indices[size - 1] = k;
                            in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                            // change to 1 if 0.
                            updates_slice_dims[size - 1] = k == 0 ? 1 : k;

                            out.slice(in_indices, in_slice_dims)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = out.slice(in_indices, in_slice_dims) +
                                              up.slice(updates_indices, updates_slice_dims);
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
            }
        }
    }
}
