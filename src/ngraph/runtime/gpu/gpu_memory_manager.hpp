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
#include <memory>
#include <stack>
#include <vector>

#include "ngraph/pass/memory_layout.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            using memory_primitive = std::function<void*(void)>;

            class GPUMemoryManager;

            class GPUAllocator
            {
            public:
                GPUAllocator() = delete;
                GPUAllocator(GPUMemoryManager* mgr)
                    : m_manager(mgr)
                {
                }

                ~GPUAllocator();
                memory_primitive reserve_argspace(void* data, size_t size);
                memory_primitive reserve_workspace(size_t size);

            private:
                GPUMemoryManager* m_manager;
                std::stack<size_t> m_active;
            };

            class GPUMemoryManager
            {
                friend class GPUAllocator;

            public:
                GPUMemoryManager();
                ~GPUMemoryManager();

                void allocate();
                GPUAllocator build_allocator() { return GPUAllocator(this); }
            private:
                size_t m_buffer_offset;
                std::vector<uint8_t> m_buffered_mem;
                pass::MemoryManager m_workspace_manager;
                static constexpr const uint16_t alignment = 4;
                void* m_argspace;
                void* m_workspace;
            };
        }
    }
}
