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

#include <cstring>

#include "ngraph/runtime/gpu/gpu_memory_manager.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace ngraph;

constexpr const uint32_t initial_buffer_size = 10 * 1024 * 1024;

runtime::gpu::GPUMemoryManager::GPUMemoryManager(GPUPrimitiveEmitter* emitter)
    : m_buffer_offset(0)
    , m_buffered_mem(initial_buffer_size, 0)
    , m_workspace_manager(new pass::MemoryManager(runtime::gpu::GPUMemoryManager::alignment))
    , m_argspace_mem(1, {nullptr, 0})
    , m_workspace_mem(1, {nullptr, 0})
    , m_primitive_emitter(emitter)
{
}

size_t runtime::gpu::GPUMemoryManager::get_allocation_size() const
{
    size_t allocation_size = 0;
    for (auto const& alloc : m_argspace_mem)
    {
        allocation_size += alloc.size;
    }
    for (auto const& alloc : m_workspace_mem)
    {
        allocation_size += alloc.size;
    }
    return allocation_size;
}

runtime::gpu::GPUMemoryManager::~GPUMemoryManager()
{
    for (auto& alloc : m_argspace_mem)
    {
        runtime::gpu::free_gpu_buffer(alloc.ptr);
    }
    for (auto& alloc : m_workspace_mem)
    {
        runtime::gpu::free_gpu_buffer(alloc.ptr);
    }
}

void runtime::gpu::GPUMemoryManager::allocate()
{
    if (m_workspace_manager->get_node_list().size() != 1)
    {
        throw std::runtime_error(
            "Attempt to allocate memory while reservations are inprogress. Ensure all "
            "GPUAllocators are closed before allocating.");
    }
    if (m_buffer_offset)
    {
        m_buffer_offset = ngraph::pass::MemoryManager::align(
            m_buffer_offset, runtime::gpu::GPUMemoryManager::alignment);
        // the back most node is always empty, fill it here
        m_argspace_mem.back().ptr = runtime::gpu::create_gpu_buffer(m_buffer_offset);
        m_argspace_mem.back().size = m_buffer_offset;
        // copy buffered kernel arguments to device
        runtime::gpu::cuda_memcpyHtD(
            m_argspace_mem.back().ptr, m_buffered_mem.data(), m_buffer_offset);
        // add an empty node to the end of the list and zero offset
        m_argspace_mem.push_back({nullptr, 0});
        m_buffered_mem.clear();
        m_buffered_mem.resize(initial_buffer_size, 0);
        m_buffer_offset = 0;
    }

    auto workspace_size = m_workspace_manager->max_allocated();
    if (workspace_size)
    {
        m_workspace_mem.back().ptr = runtime::gpu::create_gpu_buffer(workspace_size);
        m_workspace_mem.back().size = workspace_size;
        m_workspace_mem.push_back({nullptr, 0});
        m_workspace_manager.reset(
            new pass::MemoryManager(runtime::gpu::GPUMemoryManager::alignment));
    }
}

size_t runtime::gpu::GPUMemoryManager::queue_for_transfer(const void* data, size_t size)
{
    // if the current allocation will overflow the host buffer
    size_t aligned_size =
        ngraph::pass::MemoryManager::align(size, runtime::gpu::GPUMemoryManager::alignment);
    size_t new_size = m_buffer_offset + aligned_size;
    size_t buffer_size = m_buffered_mem.size();
    bool need_resize = false;
    while (buffer_size < new_size)
    {
        // add more space to the managed buffer
        buffer_size <<= 1;
        need_resize = true;
    }

    if (need_resize)
    {
        m_buffered_mem.resize(buffer_size, 0);
    }

    size_t offset = m_buffer_offset;
    std::memcpy(m_buffered_mem.data() + offset, data, size);
    m_buffer_offset += aligned_size;

    return offset;
}

runtime::gpu::GPUAllocator::GPUAllocator(GPUMemoryManager* mgr)
    : m_manager(mgr)
{
}

runtime::gpu::GPUAllocator::GPUAllocator(const GPUAllocator& g)
{
    m_manager = g.m_manager;
    m_active = g.m_active;
}

size_t runtime::gpu::GPUAllocator::reserve_argspace(const void* data, size_t size)
{
    // add parameter data to host buffer that will be transfered to device
    size_t offset = m_manager->queue_for_transfer(data, size);
    auto local = std::prev(m_manager->m_argspace_mem.end());
    // return a lambda that will yield the gpu memory address. this
    // should only be evaluated by the runtime invoked primitive
    gpu::memory_primitive mem_primitive = [=]() {
        void* argspace = (*local).ptr;
        if (argspace == nullptr)
        {
            throw std::runtime_error("An attempt was made to use unallocated device memory.");
        }
        auto gpu_mem = static_cast<uint8_t*>(argspace);
        return static_cast<void*>(gpu_mem + offset);
    };
    return m_manager->m_primitive_emitter->insert(mem_primitive);
}

size_t runtime::gpu::GPUAllocator::reserve_workspace(size_t size, bool zero_initialize)
{
    if (size == 0)
    {
        return m_manager->m_primitive_emitter->insert([]() { return nullptr; });
    }

    size_t offset = m_manager->m_workspace_manager->allocate(size);
    m_active.push(offset);
    auto local = std::prev(m_manager->m_workspace_mem.end());
    // return a lambda that will yield the gpu memory address. this
    // should only be evaluated by the runtime invoked primitive
    gpu::memory_primitive mem_primitive = [=]() {
        void* workspace = (*local).ptr;
        if (workspace == nullptr)
        {
            throw std::runtime_error("An attempt was made to use unallocated device memory.");
        }
        auto gpu_mem = static_cast<uint8_t*>(workspace);
        auto workspace_ptr = static_cast<void*>(gpu_mem + offset);
        if (zero_initialize)
        {
            runtime::gpu::cuda_memset(workspace_ptr, 0, size);
        }
        return workspace_ptr;
    };
    return m_manager->m_primitive_emitter->insert(std::move(mem_primitive));
}

void runtime::gpu::GPUAllocator::close()
{
    while (!m_active.empty())
    {
        m_manager->m_workspace_manager->free(m_active.top());
        m_active.pop();
    }
}
runtime::gpu::GPUAllocator::~GPUAllocator()
{
    this->close();
}
