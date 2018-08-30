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

#include "ngraph/axis_vector.hpp"
#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType, unsigned int InRank, unsigned int OutRank>
                void reshape(ElementType* input,
                             ElementType* output,
                             const Shape& input_shape,
                             const AxisVector& input_axis_order,
                             const Shape& output_shape)
                {
                    Eigen::array<Eigen::Index, OutRank> out_dims;
                    Eigen::array<Eigen::Index, InRank> in_dims;
                    Eigen::array<Eigen::Index, InRank> axis_order;

                    for (int i = 0; i < OutRank; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    for (int i = 0; i < InRank; i++)
                    {
                        in_dims[i] = input_shape[i];
                        axis_order[i] = input_axis_order[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, OutRank, Eigen::RowMajor>> out(
                        output, out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, InRank, Eigen::RowMajor>> in(
                        input, in_dims);

                    out.device(eigen::global_thread_pool_device) =
                        in.shuffle(axis_order).reshape(out_dims);
                }

                template <typename ElementType, unsigned int InRank, unsigned int OutRank>
                void reshape(void* input,
                             void* output,
                             const Shape& input_shape,
                             const AxisVector& input_axis_order,
                             const Shape& output_shape)
                {
                    reshape<ElementType, InRank, OutRank>(static_cast<ElementType*>(input),
                                                          static_cast<ElementType*>(output),
                                                          input_shape,
                                                          input_axis_order,
                                                          output_shape);
                }

                template <typename ElementType, unsigned int OutRank>
                void reshape_1d(void* input,
                                void* output,
                                const Shape& input_shape,
                                const AxisVector& input_axis_order,
                                const Shape& output_shape)
                {
                    reshape<ElementType, 1, OutRank>(static_cast<ElementType*>(input),
                                                     static_cast<ElementType*>(output),
                                                     input_shape,
                                                     input_axis_order,
                                                     output_shape);
                }

                template <typename ElementType, unsigned int OutRank>
                void reshape_2d(void* input,
                                void* output,
                                const Shape& input_shape,
                                const AxisVector& input_axis_order,
                                const Shape& output_shape)
                {
                    reshape<ElementType, 2, OutRank>(static_cast<ElementType*>(input),
                                                     static_cast<ElementType*>(output),
                                                     input_shape,
                                                     input_axis_order,
                                                     output_shape);
                }

                template <typename ElementType, unsigned int OutRank>
                void reshape_3d(void* input,
                                void* output,
                                const Shape& input_shape,
                                const AxisVector& input_axis_order,
                                const Shape& output_shape)
                {
                    reshape<ElementType, 3, OutRank>(static_cast<ElementType*>(input),
                                                     static_cast<ElementType*>(output),
                                                     input_shape,
                                                     input_axis_order,
                                                     output_shape);
                }

                template <typename ElementType, unsigned int OutRank>
                void reshape_4d(void* input,
                                void* output,
                                const Shape& input_shape,
                                const AxisVector& input_axis_order,
                                const Shape& output_shape)
                {
                    reshape<ElementType, 4, OutRank>(static_cast<ElementType*>(input),
                                                     static_cast<ElementType*>(output),
                                                     input_shape,
                                                     input_axis_order,
                                                     output_shape);
                }

                template <typename ElementType>
                void reshape(const void* arg,
                             void* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
                {
                    reference::reshape(static_cast<const ElementType*>(arg),
                                       static_cast<ElementType*>(out),
                                       in_shape,
                                       in_axis_order,
                                       out_shape);
                }
            }
        }
    }
}
