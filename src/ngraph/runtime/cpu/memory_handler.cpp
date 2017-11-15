// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/runtime/cpu/memory_handler.hpp"

using namespace ngraph;

runtime::cpu::MemoryHandler::MemoryHandler(size_t byte_size, size_t alignment)
    : m_allocated_buffer_pool(nullptr)
    , m_aligned_buffer_pool(nullptr)
{
    if (byte_size > 0)
    {
        size_t allocation_size = byte_size + alignment;
        m_allocated_buffer_pool = static_cast<char*>(malloc(allocation_size));
        m_aligned_buffer_pool = m_allocated_buffer_pool;
        size_t mod = size_t(m_aligned_buffer_pool) % alignment;

        if (mod != 0)
        {
            m_aligned_buffer_pool += (alignment - mod);
        }
    }
}

runtime::cpu::MemoryHandler::~MemoryHandler()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        free(m_allocated_buffer_pool);
    }
}

void* runtime::cpu::MemoryHandler::get_ptr(size_t offset) const
{
    return m_aligned_buffer_pool + offset;
}
