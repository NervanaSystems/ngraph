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
                CudaContextManager();
                ~CudaContextManager();

                CudaContextManager(CudaContextManager const&) = delete;
                CudaContextManager(CudaContextManager&&) = delete;
                CudaContextManager& operator=(CudaContextManager const&) = delete;
                CudaContextManager& operator=(CudaContextManager&&) = delete;

                CUcontext GetContext() { return m_context; }
                void SetContextCurrent() { cuCtxSetCurrent(m_context); }
            protected:
                CUdevice m_device;
                CUcontext m_context;
            };
        }
    }
}
