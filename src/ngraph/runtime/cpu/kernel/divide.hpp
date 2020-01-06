//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                typename std::enable_if<std::is_floating_point<ElementType>::value>::type
                    divide(void* input0,
                           void* input1,
                           void* output,
                           size_t count,
                           bool pythondiv,
                           int arena)
                {
                    (void)pythondiv;
                    Eigen::array<Eigen::Index, 1> out_dims, in_dims;

                    out_dims[0] = in_dims[0] = count;

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in0 / in1;
                }
                template <typename ElementType>
                typename std::enable_if<std::is_integral<ElementType>::value>::type
                    divide(void* input0,
                           void* input1,
                           void* output,
                           size_t count,
                           bool pythondiv,
                           int arena)
                {
                    Eigen::array<Eigen::Index, 1> out_dims, in_dims;

                    out_dims[0] = in_dims[0] = count;

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in_dims);
                    if (pythondiv)
                    {
                        Eigen::Tensor<ElementType, 1, Eigen::RowMajor> zero(count);
                        zero.setZero();
                        Eigen::Tensor<ElementType, 1, Eigen::RowMajor> one(count);
                        one.setConstant(1);
                        Eigen::Tensor<ElementType, 1, Eigen::RowMajor> quot = in0 / in1;
                        Eigen::Tensor<ElementType, 1, Eigen::RowMajor> rem = in0 - quot * in1;
                        Eigen::Tensor<bool, 1, Eigen::RowMajor> if_cond =
                            ((rem != zero) && ((in0 < zero) != (in1 < zero)));

                        out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = if_cond.select(quot - one, quot);
                    }
                    else
                    {
                        out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = in0 / in1;
                    }
                }
            }
        }
    }
}
