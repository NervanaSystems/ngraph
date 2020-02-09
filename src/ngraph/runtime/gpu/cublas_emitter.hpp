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

#include <cublas_v2.h>

#include "ngraph/op/dot.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUPrimitiveEmitter;

            class CUBLASEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                size_t build_dot(const element::Type& dtype,
                                 const Shape& arg0_shape,
                                 const Shape& arg1_shape,
                                 const Shape& out_shape,
                                 size_t reduction_axes,
                                 const Node* node);

                void debug_sync();
                void sync();

            private:
                CUBLASEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx);
                GPUPrimitiveEmitter* m_primitive_emitter;
                GPURuntimeContext* m_ctx;
                std::string get_error_string(std::vector<std::string>& arg_names,
                                             std::vector<Shape>& shapes,
                                             const Node* node);
            };
        }
    }
}
