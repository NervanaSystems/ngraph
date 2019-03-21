//****************************************************************************
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
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/runtime/backend.hpp"
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
        class Allocator;
        namespace cpu
        {
            class CPUAllocator;
            extern CPUAllocator& GetCPUAllocator();
        }
    }
}

class ngraph::runtime::cpu::CPUAllocator : public ngraph::runtime::Allocator
{
public:
    CPUAllocator(runtime::AllocateFunc alloc, runtime::DestroyFunc dealloc);
    ~CPUAllocator();

    void* Malloc(void* handle, size_t size, size_t alignment)
    {
        void* ptr;
        if (m_alloc)
        {
            ptr = m_alloc(handle, size, alignment);
        }
        else
        {
            ptr = malloc(size);
        }
        // check for exception
        if (size != 0 && !ptr)
        {
            throw ngraph_error("malloc failed to allocate memory of size " + std::to_string(size));
            throw std::bad_alloc();
        }
        return ptr;
    }
    void Free(void* handle, void* ptr)
    {
        if (ptr)
        {
            if (m_dealloc)
            {
                m_dealloc(handle, ptr);
            }
            else
            {
                free(ptr);
            }
        }
    }

private:
    runtime::AllocateFunc m_alloc;
    runtime::DestroyFunc m_dealloc;
};
