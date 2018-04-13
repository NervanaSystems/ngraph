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
#include <functional>
#include <unordered_map>

#include "ngraph/runtime/gpu/cuda_emitter.hpp"
#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class CUDAEmitter;
            class CUDNNEmitter;

            class GPUPrimitiveEmitter
            {
            public:
                GPUPrimitiveEmitter();
                ~GPUPrimitiveEmitter();
                std::unique_ptr<CUDAEmitter>& get_cuda_emitter();
                std::unique_ptr<CUDNNEmitter>& get_cudnn_emitter();
                std::vector<gpu::primitive*>& get_primitives() { return m_gpu_primitives; }
                size_t insert(gpu::primitive* f);
                size_t lookup(std::string hash);
                void cache(const std::string& hash, const size_t& index);

            private:
                std::unique_ptr<CUDAEmitter> m_cuda_emitter;
                std::unique_ptr<CUDNNEmitter> m_cudnn_emitter;
                std::vector<gpu::primitive*> m_gpu_primitives;
                std::unordered_map<std::string, size_t> m_primitive_map;
            };
        }
    }
}
