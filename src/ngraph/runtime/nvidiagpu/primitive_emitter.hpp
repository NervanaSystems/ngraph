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

#include "ngraph/runtime/nvidiagpu/cublas_emitter.hpp"
#include "ngraph/runtime/nvidiagpu/cuda_emitter.hpp"
#include "ngraph/runtime/nvidiagpu/cudnn_emitter.hpp"
#include "ngraph/runtime/nvidiagpu/host_emitter.hpp"
#include "ngraph/runtime/nvidiagpu/nvidiagpu_kernel_args.hpp"
#include "ngraph/runtime/nvidiagpu/nvidiagpu_memory_manager.hpp"
#include "ngraph/runtime/nvidiagpu/nvidiagpu_runtime_context.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            class PrimitiveEmitter
            {
            public:
                PrimitiveEmitter();
                PrimitiveEmitter(const std::unique_ptr<RuntimeContext>& ctx);
                std::unique_ptr<HostEmitter>& get_host_emitter();
                std::unique_ptr<CUDAEmitter>& get_cuda_emitter();
                std::unique_ptr<CUDNNEmitter>& get_cudnn_emitter();
                std::unique_ptr<CUBLASEmitter>& get_cublas_emitter();
                std::vector<nvidiagpu::primitive*>& get_primitives()
                {
                    return m_nvidiagpu_primitives;
                }
                std::vector<nvidiagpu::memory_primitive>& get_memory_primitives()
                {
                    return m_nvidiagpu_mem_primitives;
                }
                size_t insert(std::unique_ptr<nvidiagpu::primitive>&& f);
                size_t insert(const nvidiagpu::memory_primitive& f);
                size_t lookup(const std::string& hash);
                void cache(const std::string& hash, const size_t& index);
                Allocator get_memory_allocator() { return m_memory_manager.build_allocator(); }
                void allocate_primitive_memory() { m_memory_manager.allocate(); }
                size_t sizeof_device_allocation() { return m_memory_manager.get_allocation_size(); }
                KernelArgs add_kernel_args() { return KernelArgs(m_host_parameters); }
                size_t register_primitive(std::unique_ptr<nvidiagpu::primitive>&, std::string);

            private:
                std::vector<nvidiagpu::primitive*> m_nvidiagpu_primitives;
                std::vector<nvidiagpu::memory_primitive> m_nvidiagpu_mem_primitives;
                std::unordered_map<std::string, size_t> m_primitive_map;
                std::vector<std::unique_ptr<nvidiagpu::primitive>> m_managed_primitives;
                MemoryManager m_memory_manager;
                std::shared_ptr<HostParameters> m_host_parameters;
                std::unique_ptr<HostEmitter> m_host_emitter;
                std::unique_ptr<CUDAEmitter> m_cuda_emitter;
                std::unique_ptr<CUDNNEmitter> m_cudnn_emitter;
                std::unique_ptr<CUBLASEmitter> m_cublas_emitter;
            };
        }
    }
}
