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

#include <array>
#include "ngraph/codegen/code_writer.hpp"

namespace ngraph
{
    class Shape;

    namespace runtime
    {
        namespace gpu
        {
            struct GPURuntimeContext;
            class GPUPrimitiveEmitter;

            class CUDAEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                size_t build_pad(const GPURuntimeContext* ctx,
                                 const std::array<std::string, 2>& dtypes,
                                 const Shape& input_shape,
                                 const Shape& output_shape,
                                 const Shape& pad_below,
                                 const Shape& pad_above,
                                 const Shape& pad_interior,
                                 const std::string& pad_value = "");

                size_t build_1d_max_pool(const GPURuntimeContext* ctx,
                                         const std::array<std::string, 2>& dtypes,
                                         const Shape& input_shape,
                                         const Shape& output_shape,
                                         size_t window_width,
                                         size_t window_stride);

            private:
                CUDAEmitter(GPUPrimitiveEmitter* emitter);
                void print_tensor_from_gpu(codegen::CodeWriter& writer,
                                           const std::string& tensor_name,
                                           const Shape& shape);

                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
