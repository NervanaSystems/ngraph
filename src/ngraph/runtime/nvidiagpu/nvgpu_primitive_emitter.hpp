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

#pragma once

#include "ngraph/runtime/nvgpu/cublas_emitter.hpp"
#include "ngraph/runtime/nvgpu/cuda_emitter.hpp"
#include "ngraph/runtime/nvgpu/cudnn_emitter.hpp"
#include "ngraph/runtime/nvgpu/host_emitter.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_kernel_args.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_memory_manager.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_runtime_context.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvgpu
        {
            class NVPrimitiveEmitter
            {
            public:
                NVPrimitiveEmitter();
                NVPrimitiveEmitter(const std::unique_ptr<NVRuntimeContext>& ctx);
                std::unique_ptr<HostEmitter>& get_host_emitter();
                std::unique_ptr<CUDAEmitter>& get_cuda_emitter();
                std::unique_ptr<CUDNNEmitter>& get_cudnn_emitter();
                std::unique_ptr<CUBLASEmitter>& get_cublas_emitter();
                std::vector<nvgpu::primitive*>& get_primitives() { return m_nvgpu_primitives; }
                std::vector<nvgpu::memory_primitive>& get_memory_primitives()
                {
                    return m_nvgpu_mem_primitives;
                }
                size_t insert(std::unique_ptr<nvgpu::primitive>&& f);
                size_t insert(const nvgpu::memory_primitive& f);
                size_t lookup(const std::string& hash);
                void cache(const std::string& hash, const size_t& index);
                NVAllocator get_memory_allocator() { return m_memory_manager.build_allocator(); }
                void allocate_primitive_memory() { m_memory_manager.allocate(); }
                size_t sizeof_device_allocation() { return m_memory_manager.get_allocation_size(); }
                NVKernelArgs add_kernel_args() { return NVKernelArgs(m_host_parameters); }
                size_t register_primitive(std::unique_ptr<nvgpu::primitive>&, std::string);

            private:
                std::vector<nvgpu::primitive*> m_nvgpu_primitives;
                std::vector<nvgpu::memory_primitive> m_nvgpu_mem_primitives;
                std::unordered_map<std::string, size_t> m_primitive_map;
                std::vector<std::unique_ptr<nvgpu::primitive>> m_managed_primitives;
                NVMemoryManager m_memory_manager;
                std::shared_ptr<NVHostParameters> m_host_parameters;
                std::unique_ptr<HostEmitter> m_host_emitter;
                std::unique_ptr<CUDAEmitter> m_cuda_emitter;
                std::unique_ptr<CUDNNEmitter> m_cudnn_emitter;
                std::unique_ptr<CUBLASEmitter> m_cublas_emitter;
            };
        }
    }
}
