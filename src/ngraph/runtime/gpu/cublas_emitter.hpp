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

#include <functional>
#include <memory>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/axis_set.hpp"
#include "ngraph/runtime/gpu/cudnn_descriptors.hpp"
#include "ngraph/runtime/gpu/cudnn_host_parameters.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/op/dot.hpp"

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
                                 const Shape& input_tensor_shape_0,
                                 const Shape& input_tensor_shape_1,
                                 const Shape& output_tensor_shape,
                                 size_t reduction_axes);

                void debug_sync();
                void sync();

            protected:
                size_t getPrimitiveIndex(std::unique_ptr<gpu::primitive>&, std::string);

            private:
                CUBLASEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx);
                GPUPrimitiveEmitter* m_primitive_emitter;
                GPURuntimeContext* m_ctx;
            };
        } // namespace gpu
    }     // namespace runtime
} // namespace ngraph