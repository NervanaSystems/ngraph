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
#include "ngraph/runtime/reference/slice.hpp"
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
                void slice(void* input,
                           void* output,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           const Coordinate& lower_bounds,
                           int arena)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims, in_dims;
                    Eigen::array<Eigen::Index, Rank> indices;

                    for (size_t i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                        in_dims[i] = input_shape[i];
                        indices[i] = lower_bounds[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.slice(indices, out_dims);
                }

                template <typename ElementType, unsigned int Rank>
                void strided_slice(void* input,
                                   void* output,
                                   const Shape& input_shape,
                                   const Shape& output_shape,
                                   const Coordinate& lower_bounds,
                                   const Coordinate& upper_bounds,
                                   const Strides& slice_strides,
                                   int arena)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims, in_dims;
                    Eigen::array<Eigen::Index, Rank> start_indices, stop_indices, strides;

                    for (size_t i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                        in_dims[i] = input_shape[i];
                        start_indices[i] = lower_bounds[i];
                        stop_indices[i] = upper_bounds[i];
                        strides[i] = slice_strides[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.stridedSlice(start_indices, stop_indices, strides);
                }

                template <typename ElementType>
                void ref_slice(void* input,
                               void* output,
                               const Shape& input_shape,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Strides& slice_strides,
                               const Shape& output_shape)
                {
                    reference::slice<ElementType>(static_cast<const ElementType*>(input),
                                                  static_cast<ElementType*>(output),
                                                  input_shape,
                                                  lower_bounds,
                                                  upper_bounds,
                                                  slice_strides,
                                                  output_shape);
                }
            }
        }
    }
}
