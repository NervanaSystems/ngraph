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
                template <typename ElementType, unsigned int Rank1, unsigned int Rank2>
                void scatter_add(void* inputs,
                                 void* indices,
                                 void* updates,
                                 void* output,
                                 const Shape& inputs_shape,
                                 const Shape& indices_shape,
                                 const Shape& updates_shape,
                                 bool is_int64,
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

                    if (inputs != output)
                    {
                        out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = in;
                    }

                    int64_t* indices_ptr_64 = nullptr;
                    int32_t* indices_ptr_32 = nullptr;
                    if (is_int64)
                    {
                        indices_ptr_64 = static_cast<int64_t*>(indices);
                    }
                    else
                    {
                        indices_ptr_32 = static_cast<int32_t*>(indices);
                    }
                    auto size = indices_shape.size();
                    if (size == 0)
                    {
                        if (is_int64)
                        {
                            in_slice_dims[0] = in_indices[0] = indices_ptr_64[0];
                        }
                        else
                        {
                            in_slice_dims[0] = in_indices[0] = indices_ptr_32[0];
                        }
                        out.slice(in_indices, in_slice_dims)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                arena)) = out.slice(in_indices, in_slice_dims) +
                                          up.slice(updates_indices, updates_slice_dims);
                    }
                    else if (size == 1)
                    {
                        for (int i = 0; i < shape_size(indices_shape); i++)
                        {
                            if (is_int64)
                            {
                                in_slice_dims[0] = in_indices[0] = indices_ptr_64[i];
                            }
                            else
                            {
                                in_slice_dims[0] = in_indices[0] = indices_ptr_32[i];
                            }
                            in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                            updates_slice_dims[0] = updates_indices[0] = i;
                            updates_slice_dims[0] =
                                updates_slice_dims[0] == 0 ? 1 : updates_slice_dims[0];
                            out.slice(in_indices, in_slice_dims)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = out.slice(in_indices, in_slice_dims) +
                                              up.slice(updates_indices, updates_slice_dims);
                        }
                    }
                    else
                    {
                        std::vector<int> ele_counts(size);
                        ele_counts[size - 1] = 0;
                        ele_counts[size - 2] = indices_shape[size - 1];
                        for (int j = size - 3; j >= 0; j--)
                        {
                            ele_counts[j] = indices_shape[j + 2] * indices_shape[j + 1];
                        }
                        for (int i = 0; i < shape_size(indices_shape); i++)
                        {
                            if (is_int64)
                            {
                                in_slice_dims[0] = in_indices[0] = indices_ptr_64[i];
                            }
                            else
                            {
                                in_slice_dims[0] = in_indices[0] = indices_ptr_32[i];
                            }
                            int k = i;
                            for (int j = 0; j < size - 1; j++)
                            {
                                updates_slice_dims[j] = updates_indices[j] = k / ele_counts[j];
                                updates_slice_dims[j] =
                                    updates_slice_dims[j] == 0 ? 1 : updates_slice_dims[j];
                                k = k % ele_counts[j];
                            }
                            updates_indices[size - 1] = k;
                            in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                            updates_slice_dims[size - 1] = k == 0 ? 1 : k;

                            out.slice(in_indices, in_slice_dims)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = out.slice(in_indices, in_slice_dims) +
                                              up.slice(updates_indices, updates_slice_dims);
                        }
                    }
                }
            }
        }
    }
}
