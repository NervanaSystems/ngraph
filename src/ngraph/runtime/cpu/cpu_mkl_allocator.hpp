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
#include <cstdint>
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

namespace mkl
{
    extern "C" {
    typedef void* (*i_malloc_t)(size_t size);
    typedef void* (*i_calloc_t)(size_t nmemb, size_t size);
    typedef void* (*i_realloc_t)(void* ptr, size_t size);
    typedef void (*i_free_t)(void* ptr);

    extern i_malloc_t i_malloc;
    extern i_calloc_t i_calloc;
    extern i_realloc_t i_realloc;
    extern i_free_t i_free;
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPUAllocator;
        }
    }
}

class ngraph::runtime::cpu::CPUAllocator
{
public:
    CPUAllocator(size_t size);
    CPUAllocator();
    ~CPUAllocator();

private:
    void* m_buffer;
    size_t m_byte_size;

    static inline void* MallocHook(size_t size)
    {
        void* ptr = static_cast<void*>(ngraph_malloc(size));
        return ptr;
    }
    static inline void FreeHook(void* ptr) { ngraph_free(ptr); }
};
