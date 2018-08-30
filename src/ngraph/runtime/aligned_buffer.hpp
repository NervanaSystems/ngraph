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

#pragma once

#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        class AlignedBuffer;
    }
}

/// \brief Allocates a block of memory on the specified alignment. The actual size of the
/// allocated memory is larger than the requested size by the alignment, so allocating 1 byte
/// on 64 byte alignment will allocate 65 bytes.
class ngraph::runtime::AlignedBuffer
{
public:
    AlignedBuffer(size_t byte_size, size_t alignment);
    AlignedBuffer();
    void initialize(size_t byte_size, size_t alignment);
    ~AlignedBuffer();

    size_t size() const { return m_byte_size; }
    void* get_ptr(size_t offset) const { return m_aligned_buffer + offset; }
    void* get_ptr() const { return m_aligned_buffer; }
private:
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    char* m_allocated_buffer;
    char* m_aligned_buffer;
    size_t m_byte_size;
};
