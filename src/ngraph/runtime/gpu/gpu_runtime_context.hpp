/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <string>
#include <unordered_map>

#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            typedef std::function<void(void**, void**)> primitive;

            extern "C" {
            struct GPURuntimeContext
            {
                cudnnHandle_t* cudnn_handle;
                cublasHandle_t* cublas_handle;
                gpu::primitive* const* gpu_primitives;
                CudaFunctionPool* compiled_kernel_pool;
                // Note that in it's current state, calling methods of CudaFunctionPool
                // or other native compiled C++ functions in ngraph from the JIT code is
                // unsafe and will fail if the GLIBCXX versions are diffent for the
                // native compiler and clang. If all of the emitted CUDA ops are refactored
                // to use the GPUPrimitiveEmitter, the above pointer can be removed. It is left now
                // for backward compatability.
            };
            }
        }
    }
}
