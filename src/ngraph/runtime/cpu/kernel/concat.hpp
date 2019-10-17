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
#include <functional>
#include <iostream>
#include <vector>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

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
                template <typename ElementType, int Rank>
                void concat(std::vector<void*> inputs,
                            std::vector<Shape> input_shapes,
                            void* output,
                            Shape output_shape,
                            int64_t axis)
                {
                    Eigen::array<Eigen::Index, Rank> out_dims;
                    for (int i = 0; i < Rank; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);

                    Eigen::array<Eigen::Index, Rank> in_dims, concat_pos;
                    concat_pos.fill(static_cast<Eigen::Index>(0));
                    for (size_t i = 0; i < input_shapes.size(); i++)
                    {
                        for (int j = 0; j < Rank; j++)
                        {
                            in_dims[j] = input_shapes[i][j];
                        }

                        Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                            static_cast<ElementType*>(inputs[i]), in_dims);
                        out.slice(concat_pos, in_dims)
                            .device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                                0)) = in;
                        concat_pos[axis] += in_dims[axis];
                    }
                }
            }
        }
    }
}
