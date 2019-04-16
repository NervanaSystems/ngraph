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

#include <algorithm>
#include <memory>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::AlignedBuffer::AlignedBuffer()
    : m_allocator(nullptr)
    , m_allocated_buffer(nullptr)
    , m_aligned_buffer(nullptr)
    , m_byte_size(0)
{
}

runtime::AlignedBuffer::AlignedBuffer(size_t byte_size, size_t alignment, Allocator* allocator)
    : m_allocator(allocator)
    , m_byte_size(byte_size)
{
    m_byte_size = std::max<size_t>(1, byte_size);
    size_t allocation_size = m_byte_size + alignment;
    m_allocated_buffer = static_cast<char*>(m_allocator->Malloc(m_byte_size, alignment));
    m_aligned_buffer = m_allocated_buffer;
    size_t mod = size_t(m_aligned_buffer) % alignment;

    if (mod != 0)
    {
        m_aligned_buffer += (alignment - mod);
    }
}

runtime::AlignedBuffer::~AlignedBuffer()
{
    if (m_allocated_buffer != nullptr)
    {
        m_allocator->Free(m_allocated_buffer);
    }
}
