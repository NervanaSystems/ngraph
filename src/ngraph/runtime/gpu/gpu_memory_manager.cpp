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
#include <cstring>

#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/gpu_memory_manager.hpp"


using namespace ngraph;

constexpr const uint32_t initial_buffer_size = 10*1024*1024;

runtime::gpu::GPUMemoryManager::GPUMemoryManager()
    : m_buffer_offset(0), m_buffered_mem(initial_buffer_size), m_workspace_manager(alignment),
      m_argspace(nullptr), m_workspace(nullptr)
{
}
runtime::gpu::GPUMemoryManager::~GPUMemoryManager()
{
    runtime::gpu::free_gpu_buffer(m_argspace);
    runtime::gpu::free_gpu_buffer(m_workspace);
}
void runtime::gpu::GPUMemoryManager::allocate()
{
    if (m_buffer_offset)
    {
        m_argspace = runtime::gpu::create_gpu_buffer(m_buffer_offset);
        runtime::gpu::cuda_memcpyHtD(m_argspace, m_buffered_mem.data(), m_buffer_offset);
    }
    auto workspace_size = m_workspace_manager.max_allocated();
    if (workspace_size)
    {
        m_workspace = runtime::gpu::create_gpu_buffer(workspace_size);
    }
}
runtime::gpu::memory_primitive runtime::gpu::GPUAllocator::reserve_argspace(void* data, size_t size)
{
    // if the current allocation will overflow the host buffer
    if (m_manager->m_buffer_offset + size > m_manager->m_buffered_mem.size())
    {
        // add more space to the managed buffer
        size_t new_size = m_manager->m_buffered_mem.size()/initial_buffer_size + 1;
        m_manager->m_buffered_mem.resize(new_size);
    }

    size_t offset = m_manager->m_buffer_offset;
    std::memcpy(m_manager->m_buffered_mem.data() + offset, data, size);
    m_manager->m_buffer_offset += size;

    // return a lambda that will yield the gpu memory address. this
    // should only be evaluated by the runtime invoked primitive
    auto manager = m_manager;
    return [=](){
        auto gpu_mem = static_cast<uint8_t*>(manager->m_argspace);
        return  static_cast<void*>(gpu_mem + offset);
    };
}
runtime::gpu::memory_primitive runtime::gpu::GPUAllocator::reserve_workspace(size_t size)
{
    size_t offset = m_manager->m_workspace_manager.allocate(size);
    m_active.push(offset);
    // return a lambda that will yield the gpu memory address. this
    // should only be evaluated by the runtime invoked primitive
    return [&, offset](){
        auto gpu_mem = static_cast<uint8_t*>(m_manager->m_workspace);
        return  static_cast<void*>(gpu_mem + offset);
    };
}
runtime::gpu::GPUAllocator::~GPUAllocator()
{
    while(!m_active.empty())
    {
        m_manager->m_workspace_manager.free(m_active.top());
        m_active.pop();
    }
}
