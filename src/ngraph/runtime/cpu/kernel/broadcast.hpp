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

#include "ngraph/axis_set.hpp"
#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
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
                void broadcast(void* input,
                               void* output,
                               const Shape& input_shape,
                               const Shape& output_shape)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims;
                    Eigen::array<Eigen::Index, Rank> in_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                        in_dims[i] = input_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);

                    Eigen::array<ptrdiff_t, Rank> factors;
                    for (int i = 0; i < Rank; i++)
                    {
                        factors[i] = output_shape[i] / input_shape[i];
                    }

                    out.device(eigen::global_thread_pool_device) = in.broadcast(factors);
                }
            }
        }
    }
}
