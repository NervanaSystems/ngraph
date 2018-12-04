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
                template <typename ElementType, unsigned int Rank>
                void replace_slice(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Coordinate& lower_bounds,
                                   int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in0_dims, in1_dims;
                    Eigen::array<Eigen::Index, Rank> indices;

                    for (int i = 0; i < Rank; i++)
                    {
                        in0_dims[i] = input0_shape[i];
                        in1_dims[i] = input1_shape[i];
                        indices[i] = lower_bounds[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), in0_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in0_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in1_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in0;
                    out.slice(indices, in1_dims)
                        .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = in1;
                }

                template <typename ElementType, unsigned int Rank>
                void strided_replace_slice(void* input0,
                                           void* input1,
                                           void* output,
                                           const Shape& input0_shape,
                                           const Shape& input1_shape,
                                           const Coordinate& lower_bounds,
                                           const Coordinate& upper_bounds,
                                           const Strides& slice_strides,
                                           int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in0_dims, in1_dims;
                    Eigen::array<Eigen::Index, Rank> start_indices, stop_indices, strides;

                    for (int i = 0; i < Rank; i++)
                    {
                        in0_dims[i] = input0_shape[i];
                        in1_dims[i] = input1_shape[i];
                        start_indices[i] = lower_bounds[i];
                        stop_indices[i] = upper_bounds[i];
                        strides[i] = slice_strides[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), in0_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in0_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in1_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in0;
                    out.stridedSlice(start_indices, stop_indices, strides)
                        .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = in1;
                }
            }
        }
    }
}
