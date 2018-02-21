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

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class Cuda_context_manager
            {
                public:
                static Cuda_context_manager& Instance()
                {
                    static Cuda_context_manager manager;
                    return pool;
                }

                Cuda_context_manager(Cuda_context_manager const&) = delete;
                Cuda_context_manager(Cuda_context_manager&&) = delete;
                Cuda_context_manager& operator=(Cuda_context_manager const&) = delete;
                Cuda_context_manager& operator=(Cuda_context_manager &&) = delete;

                std::shared_ptr<CUcontext> GetContext()
                {
                    return context_ptr;
                }

                protected:
                Cuda_context_manager()
                {
                    CUDA_SAFE_CALL(cuInit(0));
                    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
                    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
                    context_ptr = std::make_shared<CUcontext>(context);
                }
                ~Cuda_context_manager(){}
                CUdevice cuDevice;
                CUcontext context;
                std::shared_ptr<CUcontext> context_ptr;
            }
        }
    }
}
