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

#include <list>
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
            class GPUPrimitiveEmitter;
            class GPUMemoryManager;

            class GPUAllocator
            {
            public:
                GPUAllocator() = delete;
                GPUAllocator(GPUMemoryManager* mgr);
                GPUAllocator(const GPUAllocator& g);

                ~GPUAllocator();
                template <typename T>
                size_t reserve_argspace(const T& container)
                {
                    return reserve_argspace(container.data(),
                                            container.size() * sizeof(typename T::value_type));
                }
                size_t reserve_argspace(const void* data, size_t size);
                size_t reserve_workspace(size_t size, bool zero_initialize = true);

                void close();

            private:
                GPUMemoryManager* m_manager;
                std::stack<size_t> m_active;
            };

            class GPUMemoryManager
            {
                friend class GPUPrimitiveEmitter;
                friend class GPUAllocator;

            public:
                ~GPUMemoryManager();

                void allocate();
                size_t get_allocation_size() const;
                GPUAllocator build_allocator() { return GPUAllocator(this); }
            private:
                GPUMemoryManager(GPUPrimitiveEmitter* emitter);
                size_t queue_for_transfer(const void* data, size_t size);

                size_t m_buffer_offset;
                std::vector<uint8_t> m_buffered_mem;
                ngraph::pass::MemoryManager m_workspace_manager;
                static constexpr const uint16_t alignment = 8;

                struct allocation
                {
                    void* ptr;
                    size_t size;
                };

                std::list<allocation> m_argspace_mem;
                std::list<allocation> m_workspace_mem;
                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
