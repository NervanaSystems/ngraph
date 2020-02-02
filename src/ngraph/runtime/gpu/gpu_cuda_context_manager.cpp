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

#include <memory>
#include <string>

#include "ngraph/runtime/gpu/cuda_error_check.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"

using namespace ngraph;

runtime::gpu::CudaContextManager::CudaContextManager()
{
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&m_device, 0));
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&m_context, m_device));
}

runtime::gpu::CudaContextManager::~CudaContextManager()
{
    CUDA_SAFE_CALL_NO_THROW(cuDevicePrimaryCtxRelease(m_device));
}
