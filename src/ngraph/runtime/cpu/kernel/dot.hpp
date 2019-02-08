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

#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/reference/dot.hpp"
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
                          unsigned int Input0Rank,
                          unsigned int Input1Rank,
                          unsigned int DotDims>
                void dot(void* input0,
                         void* input1,
                         void* output,
                         const Shape& input0_shape,
                         const Shape& input1_shape,
                         const Shape& output_shape,
                         int arena)
                {
                    constexpr unsigned int OutRank = Input0Rank + Input1Rank - 2 * DotDims;

                    Eigen::array<Eigen::Index, OutRank> out_dims;
                    Eigen::array<Eigen::Index, Input0Rank> in0_dims;
                    Eigen::array<Eigen::Index, Input1Rank> in1_dims;
                    Eigen::array<Eigen::IndexPair<Eigen::Index>, DotDims> dot_dims;

                    for (int i = 0; i < OutRank; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    for (int i = 0; i < Input0Rank; i++)
                    {
                        in0_dims[i] = input0_shape[i];
                    }

                    for (int i = 0; i < Input1Rank; i++)
                    {
                        in1_dims[i] = input1_shape[i];
                    }

                    for (int i = 0; i < DotDims; i++)
                    {
                        dot_dims[i].first = Input0Rank - DotDims + i;
                        dot_dims[i].second = i;
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, OutRank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Input0Rank, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in0_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Input1Rank, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in1_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in0.contract(in1, dot_dims);
                }

                template <typename ElementType>
                void dot_scalar(
                    void* input0, void* input1, void* output, size_t element_count, int arena)
                {
                    Eigen::array<Eigen::Index, 1> out_dims;
                    Eigen::array<Eigen::Index, 1> in1_dims;

                    out_dims[0] = element_count;
                    in1_dims[0] = element_count;

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    auto in0 = static_cast<ElementType*>(input0);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in1_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in0[0] * in1;
                }

                template <typename ElementType>
                void dot_1d_1d_1rd(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape,
                                   int arena)
                {
                    dot<ElementType, 1, 1, 1>(
                        input0, input1, output, input0_shape, input1_shape, output_shape, arena);
                }

                template <typename ElementType>
                void dot_2d_1d_1rd(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape,
                                   int arena)
                {
                    dot<ElementType, 2, 1, 1>(
                        input0, input1, output, input0_shape, input1_shape, output_shape, arena);
                }

                template <typename ElementType>
                void dot_1d_2d_1rd(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape,
                                   int arena)
                {
                    dot<ElementType, 1, 2, 1>(
                        input0, input1, output, input0_shape, input1_shape, output_shape, arena);
                }

                template <typename ElementType>
                void dot_3d_3d_1rd(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape,
                                   int arena)
                {
                    dot<ElementType, 3, 3, 1>(
                        input0, input1, output, input0_shape, input1_shape, output_shape, arena);
                }

                template <typename ElementType>
                void dot_3d_2d_1rd(void* input0,
                                   void* input1,
                                   void* output,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape,
                                   int arena)
                {
                    dot<ElementType, 3, 2, 1>(
                        input0, input1, output, input0_shape, input1_shape, output_shape, arena);
                }

                template <typename ElementType>
                void dot_ref(void* arg0,
                             void* arg1,
                             void* out,
                             const Shape& arg0_shape,
                             const Shape& arg1_shape,
                             const Shape& out_shape,
                             size_t reduction_axes_count)
                {
                    reference::dot(static_cast<const ElementType*>(arg0),
                                   static_cast<const ElementType*>(arg1),
                                   static_cast<ElementType*>(out),
                                   arg0_shape,
                                   arg1_shape,
                                   out_shape,
                                   reduction_axes_count);
                }
            }
        }
    }
}
