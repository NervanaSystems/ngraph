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

#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            extern "C" {
            struct GPURuntimeContext
            {
                CudaFunctionPool* nvrtc_cache;
                //CudaContextManager* cuda_manager;

                GPURuntimeContext()
                    : nvrtc_cache(new CudaFunctionPool)
                    {
                        // Create context use driver API and make it current, the runtime call will pickup the context
                        // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interoperability-between-runtime-and-driver-apis
                        ngraph::runtime::gpu::CudaContextManager::instance();
                    }
                ~GPURuntimeContext()
                {
                    delete nvrtc_cache;
                }
            };
            }
        }
    }
}
