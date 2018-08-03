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
#include "ngraph/runtime/reference/relu.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void relu(void* input0, void* output, size_t count)
                {
                    Eigen::array<Eigen::Index, 1> out_dims, in_dims;

                    out_dims[0] = in_dims[0] = count;

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in_dims);

                    out.device(eigen::global_thread_pool_device) = in0.cwiseMax(ElementType(0));
                }

                template <typename ElementType>
                void relu_backprop(void* arg, void* delta_arg, void* out, size_t count)
                {
                    reference::relu_backprop<ElementType>(static_cast<ElementType*>(arg),
                                                          static_cast<ElementType*>(delta_arg),
                                                          static_cast<ElementType*>(out),
                                                          count);
                }
            }
        }
    }
}
