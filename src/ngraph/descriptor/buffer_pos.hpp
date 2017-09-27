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

#pragma once

#include <cassert>

#include "ngraph/descriptor/buffer.hpp"

namespace ngraph
{
    namespace descriptor
    {
        /// @brief Specifies a contiguous portion of a buffer.
        ///
        /// Currently only implemented for linear buffers.
        class BufferPos
        {
        public:
            BufferPos() {}

            BufferPos(std::shared_ptr<Buffer> buffer, size_t offset, size_t size)
                : m_buffer(buffer)
                , m_offset(offset)
                , m_size(size)
            {
                assert(buffer->size() >= offset + size);
            }

            BufferPos(const BufferPos& buffer_pos) = default;
            BufferPos& operator=(const BufferPos& buffer_pos) = default;

        protected:
            std::shared_ptr<Buffer> m_buffer;
            size_t                  m_offset;
            size_t                  m_size;
        };
    }
}
