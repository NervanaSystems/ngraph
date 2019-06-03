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
                    auto size = indices_shape.size();

                    if (size == 0)
                    {
                        Eigen::array<Eigen::Index, Rank1> in_slice_dims, in_indices;
                        Eigen::array<Eigen::Index, Rank2> out_slice_dims, out_indices;

                        for (int i = 0; i < Rank1; i++)
                        {
                            in_slice_dims[i] = inputs_shape[i];
                            in_indices[i] = 0;
                        }
                        for (int i = 0; i < Rank2; i++)
                        {
                            out_slice_dims[i] = output_shape[i];
                            out_indices[i] = 0;
                        }

                        in_slice_dims[0] = in_indices[0] = indices_ptr[0];
                        // change to 1 if 0.
                        in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                        out.slice(out_indices, out_slice_dims)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                arena)) = in.slice(in_indices, in_slice_dims);
                    }
                    else
                    {
                        // This is used to calculate the leading indices for out_indices and out_slice_dims
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

                        //#pragma omp parallel for
                        for (int i = 0; i < shape_size(indices_shape); i++)
                        {
                            // Declare these inside the loop for omp parallel
                            Eigen::array<Eigen::Index, Rank1> in_slice_dims, in_indices;
                            Eigen::array<Eigen::Index, Rank2> out_slice_dims, out_indices;

                            for (int i = 0; i < Rank1; i++)
                            {
                                in_slice_dims[i] = inputs_shape[i];
                                in_indices[i] = 0;
                            }
                            for (int i = 0; i < Rank2; i++)
                            {
                                out_slice_dims[i] = output_shape[i];
                                out_indices[i] = 0;
                            }

                            in_slice_dims[0] = in_indices[0] = indices_ptr[i];
                            int k = i;
                            for (int j = 0; j < size - 1; j++)
                            {
                                out_slice_dims[j] = out_indices[j] = k / ele_counts[j];
                                // change to 1 if 0.
                                out_slice_dims[j] = out_slice_dims[j] == 0 ? 1 : out_slice_dims[j];
                                k = k % ele_counts[j];
                            }
                            out_indices[size - 1] = k;
                            in_slice_dims[0] = in_slice_dims[0] == 0 ? 1 : in_slice_dims[0];
                            // change to 1 if 0.
                            out_slice_dims[size - 1] = k == 0 ? 1 : k;

                            out.slice(out_indices, out_slice_dims)
                                .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                    arena)) = in.slice(in_indices, in_slice_dims);
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
