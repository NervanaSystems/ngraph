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

#include "ngraph/runtime/cpu/cpu_executor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename InputElementType, typename OutputElementType>
                void convert(void* input, void* output, size_t count, int arena)
                {
                    Eigen::array<Eigen::Index, 1> out_dims, in_dims;

                    out_dims[0] = in_dims[0] = count;

                    Eigen::TensorMap<Eigen::Tensor<OutputElementType, 1, Eigen::RowMajor>> out(
                        static_cast<OutputElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<InputElementType, 1, Eigen::RowMajor>> in(
                        static_cast<InputElementType*>(input), in_dims);

                    out.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) =
                        in.template cast<OutputElementType>();
                }

                template <typename InputElementType>
                void convert_to_float32(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, float>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_float64(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, double>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_i8(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, int8_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_i16(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, int16_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_i32(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, int32_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_i64(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, int64_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_u8(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, uint8_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_u16(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, uint16_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_u32(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, uint32_t>(input, output, count, arena);
                }

                template <typename InputElementType>
                void convert_to_u64(void* input, void* output, size_t count, int arena)
                {
                    convert<InputElementType, uint64_t>(input, output, count, arena);
                }
            }
        }
    }
}
