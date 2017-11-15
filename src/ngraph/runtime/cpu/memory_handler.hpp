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

#include <cstddef>
#include <memory>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class MemoryHandler;
        }
    }
}

class ngraph::runtime::cpu::MemoryHandler
{
public:
    MemoryHandler(size_t pool_size, size_t alignment);
    ~MemoryHandler();

    void* get_ptr(size_t offset) const;

private:
    char* m_allocated_buffer_pool;
    char* m_aligned_buffer_pool;
};
