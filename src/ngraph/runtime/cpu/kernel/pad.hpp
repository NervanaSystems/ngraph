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
#include "ngraph/runtime/reference/pad.hpp"
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
                void pad(void* input,
                         void* output,
                         void* pad_value,
                         const Shape& input_shape,
                         const Shape& output_shape,
                         const CoordinateDiff& padding_below,
                         const CoordinateDiff& padding_above,
                         int arena)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims, in_dims;
                    Eigen::array<Eigen::IndexPair<size_t>, Rank> padding;

                    for (int i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                        in_dims[i] = input_shape[i];
                        padding[i] = {padding_below[i], padding_above[i]};
                    }
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.pad(padding, *static_cast<ElementType*>(pad_value));
                }

                template <typename ElementType, unsigned int Rank>
                void pad_and_slice(void* input,
                                   void* output,
                                   void* pad_value,
                                   const Shape& input_shape,
                                   const Shape& output_shape,
                                   const CoordinateDiff& padding_below,
                                   const CoordinateDiff& padding_above,
                                   int arena)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims, in_dims;
                    Eigen::array<Eigen::IndexPair<size_t>, Rank> padding;
                    Eigen::array<Eigen::Index, Rank> indices;

                    for (size_t i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                        in_dims[i] = input_shape[i];

                        padding[i] = {
                            padding_below[i] >= 0 ? static_cast<unsigned long int>(padding_below[i])
                                                  : 0,
                            padding_above[i] >= 0 ? static_cast<unsigned long int>(padding_above[i])
                                                  : 0};

                        if (padding_below[i] < 0)
                        {
                            NGRAPH_CHECK(padding_below[i] > INT_MIN);
                            indices[i] = -padding_below[i];
                        }
                        else
                        {
                            indices[i] = 0;
                        }
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.pad(padding, *static_cast<ElementType*>(pad_value))
                            .slice(indices, out_dims);
                }

                template <typename ElementType>
                void pad_ref(const void* arg0,
                             const void* arg1,
                             void* out,
                             const Shape& arg0_shape,
                             const Shape& out_shape,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above,
                             const ngraph::op::PadMode pad_mode,
                             int /* arena */)
                {
                    reference::pad(static_cast<const ElementType*>(arg0),
                                   static_cast<const ElementType*>(arg1),
                                   static_cast<ElementType*>(out),
                                   arg0_shape,
                                   out_shape,
                                   padding_below,
                                   padding_above,
                                   pad_mode);
                }
            }
        }
    }
}
