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

#include "ngraph/axis_set.hpp"
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
                void softmax_all(void* input, void* output, const Shape& input_shape, int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims, rdims;
                    rdims.fill(1);

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType *>(output), in_dims),
                        in(static_cast<ElementType *>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        (in - in.maximum().eval().reshape(rdims).broadcast(in_dims)).exp();
                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        out * out.sum().inverse().eval().reshape(rdims).broadcast(in_dims);
                }

                template <typename ElementType, unsigned int Rank, unsigned int AxisCount>
                void softmax(void* input,
                             void* output,
                             const Shape& input_shape,
                             const AxisSet& softmax_axes,
                             int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims, rdims, bcast;
                    Eigen::array<Eigen::Index, AxisCount> axes;
                    rdims.fill(1);

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank; i++)
                    {
                        if (softmax_axes.count(i))
                        {
                            rdims[i] = 1;
                        }
                        else
                        {
                            rdims[i] = in_dims[i];
                        }
                    }
                    for (int i = 0; i < Rank; i++)
                    {
                        bcast[i] = in_dims[i] / rdims[i];
                    }

                    int i = 0;
                    for (auto axis : softmax_axes)
                    {
                        axes[i++] = axis;
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType *>(output), in_dims),
                        in(static_cast<ElementType *>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        (in - in.maximum(axes).eval().reshape(rdims).broadcast(bcast)).exp();
                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        out * out.sum(axes).inverse().eval().reshape(rdims).broadcast(bcast);
                }

                template <typename ElementType, unsigned int Rank>
                void softmax_innermost_1rd(void* input,
                                           void* output,
                                           const Shape& input_shape,
                                           int arena)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims, rdims, bcast;
                    Eigen::IndexList<Eigen::type2index<Rank - 1>> axis;
                    rdims.fill(1);

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank - 1; i++)
                    {
                        rdims[i] = in_dims[i];
                    }

                    for (int i = 0; i < Rank; i++)
                    {
                        bcast[i] = in_dims[i] / rdims[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType *>(output), in_dims),
                        in(static_cast<ElementType *>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        (in - in.maximum(axis).eval().reshape(rdims).broadcast(bcast)).exp();
                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        out * out.sum(axis).inverse().eval().reshape(rdims).broadcast(bcast);
                }

                template <typename ElementType, unsigned int Rank>
                void softmax_1rd(void* input,
                                 void* output,
                                 const Shape& input_shape,
                                 const AxisSet& softmax_axes,
                                 int arena)
                {
                    softmax<ElementType, Rank, 1>(input, output, input_shape, softmax_axes, arena);
                }

                template <typename ElementType>
                void softmax_3d_2rd(void* input,
                                    void* output,
                                    const Shape& input_shape,
                                    const AxisSet& softmax_axes,
                                    int arena)
                {
                    softmax<ElementType, 3, 2>(input, output, input_shape, softmax_axes, arena);
                }

                template <typename ElementType>
                void softmax_4d_3rd(void* input,
                                    void* output,
                                    const Shape& input_shape,
                                    const AxisSet& softmax_axes,
                                    int arena)
                {
                    softmax<ElementType, 4, 3>(input, output, input_shape, softmax_axes, arena);
                }
            }
        }
    }
}
