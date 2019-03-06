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

#include <cstddef>
#include "ngraph/runtime/cpu/cpu_mkl_allocator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPUAlignedBuffer;
        }
    }
}

/// \brief Allocates a block of memory on the specified alignment. The actual size of the
/// allocated memory is larger than the requested size by the alignment, so allocating 1 byte
/// on 64 byte alignment will allocate 65 bytes.
class ngraph::runtime::cpu::CPUAlignedBuffer
{
public:
    CPUAlignedBuffer(size_t byte_size,
                     size_t alignment,
                     ngraph::runtime::cpu::CPUAllocator* cpu_allocator);
    ~CPUAlignedBuffer();

    size_t size() const { return m_byte_size; }
    void* get_ptr(size_t offset) const { return m_aligned_buffer + offset; }
    void* get_ptr() const { return m_aligned_buffer; }
private:
    CPUAlignedBuffer(const CPUAlignedBuffer&) = delete;
    CPUAlignedBuffer(CPUAlignedBuffer&&) = delete;
    CPUAlignedBuffer& operator=(const CPUAlignedBuffer&) = delete;

    ngraph::runtime::cpu::CPUAllocator* m_cpu_allocator;
    char* m_allocated_buffer;
    char* m_aligned_buffer;
    size_t m_byte_size;
};
