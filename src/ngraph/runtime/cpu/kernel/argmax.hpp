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
                template <typename InType, typename OutType, unsigned int Rank>
                void argmax(void* input,
                            void* output,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            size_t axis,
                            int arena)
                {
                    Eigen::array<Eigen::Index, Rank - 1> out_dims;
                    Eigen::array<Eigen::Index, Rank> in_dims;

                    for (size_t i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (size_t i = 0; i < Rank - 1; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<OutType, Rank - 1, Eigen::RowMajor>> out(
                        static_cast<OutType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<InType, Rank, Eigen::RowMajor>> in(
                        static_cast<InType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.argmax(axis).template cast<OutType>();
                }
            }
        }
    }
}
