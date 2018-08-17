/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
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
                void reduce_min_all(void* input,
                                    void* output,
                                    const Shape& input_shape,
                                    const Shape& output_shape)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;
                    Eigen::array<Eigen::Index, 0> out_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 0, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);
                    out.device(eigen::global_thread_pool_device) = in.minimum();
                }

                template <typename ElementType, unsigned int Rank>
                void reduce_min_innermost_1rd(void* input,
                                              void* output,
                                              const Shape& input_shape,
                                              const Shape& output_shape)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;
                    Eigen::array<Eigen::Index, Rank - 1> out_dims;
                    Eigen::IndexList<Eigen::type2index<Rank - 1>> reduction_dim;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank - 1; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank - 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);
                    out.device(eigen::global_thread_pool_device) = in.minimum(reduction_dim);
                }

                template <typename ElementType, unsigned int Rank, unsigned int ReductionDims>
                void reduce_min(void* input,
                                void* output,
                                const Shape& input_shape,
                                const Shape& output_shape,
                                const AxisSet& reduction_axes)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;
                    Eigen::array<Eigen::Index, Rank - ReductionDims> out_dims;
                    Eigen::array<Eigen::Index, ReductionDims> reduction_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank - ReductionDims; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    int i = 0;
                    for (auto axis : reduction_axes)
                    {
                        reduction_dims[i++] = axis;
                    }

                    Eigen::TensorMap<
                        Eigen::Tensor<ElementType, Rank - ReductionDims, Eigen::RowMajor>>
                        out(static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);
                    out.device(eigen::global_thread_pool_device) = in.minimum(reduction_dims);
                }

                template <typename ElementType, unsigned int Rank>
                void reduce_min_1rd(void* input,
                                    void* output,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    const AxisSet& reduction_axes)
                {
                    reduce_min<ElementType, Rank, 1>(
                        input, output, input_shape, output_shape, reduction_axes);
                }

                template <typename ElementType>
                void reduce_min_3d_2rd(void* input,
                                       void* output,
                                       const Shape& input_shape,
                                       const Shape& output_shape,
                                       const AxisSet& reduction_axes)
                {
                    reduce_min<ElementType, 3, 2>(
                        input, output, input_shape, output_shape, reduction_axes);
                }

                template <typename ElementType>
                void reduce_min_4d_2rd(void* input,
                                       void* output,
                                       const Shape& input_shape,
                                       const Shape& output_shape,
                                       const AxisSet& reduction_axes)
                {
                    reduce_min<ElementType, 4, 2>(
                        input, output, input_shape, output_shape, reduction_axes);
                }

                template <typename ElementType>
                void reduce_min_5d_2rd(void* input,
                                       void* output,
                                       const Shape& input_shape,
                                       const Shape& output_shape,
                                       const AxisSet& reduction_axes)
                {
                    reduce_min<ElementType, 5, 2>(
                        input, output, input_shape, output_shape, reduction_axes);
                }

                template <typename ElementType>
                void min(void* arg,
                         void* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         const AxisSet& reduction_axes)
                {
                    reference::min(static_cast<ElementType*>(arg),
                                   static_cast<ElementType*>(out),
                                   in_shape,
                                   out_shape,
                                   reduction_axes);
                }
            }
        }
    }
}
