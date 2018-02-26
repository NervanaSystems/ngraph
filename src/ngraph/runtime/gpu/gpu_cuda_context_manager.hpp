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

#include <memory>
#include <string>

#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class CudaContextManager
            {
            public:
                static CudaContextManager& instance()
                {
                    static CudaContextManager manager;
                    return manager;
                }

                CudaContextManager(CudaContextManager const&) = delete;
                CudaContextManager(CudaContextManager&&) = delete;
                CudaContextManager& operator=(CudaContextManager const&) = delete;
                CudaContextManager& operator=(CudaContextManager&&) = delete;

                std::shared_ptr<CUcontext> get_context() { return m_context_ptr; }
            protected:
                CudaContextManager()
                {
                    CUDA_SAFE_CALL(cuInit(0));
                    CUDA_SAFE_CALL(cuDeviceGet(&m_device, 0));
                    CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, m_device));
                    m_context_ptr = std::make_shared<CUcontext>(m_context);
                }
                ~CudaContextManager() {}
                CUdevice m_device;
                CUcontext m_context;
                std::shared_ptr<CUcontext> m_context_ptr;
            };
        }
    }
}
