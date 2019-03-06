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
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);

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
        class SystemAllocator;
        class FrameworkAllocator;
        namespace cpu
        {
            class CPUAllocator;
        }
    }
}

// This class will be instantiated in the CPUCallFrame with the allocator object which will be used for the
// device memory allocation
class ngraph::runtime::cpu::CPUAllocator
{
public:
    CPUAllocator(ngraph::runtime::Allocator* alloctor, size_t alignment);
    ~CPUAllocator();

    void* malloc(size_t size);
    void free(void* ptr);

private:
    std::unique_ptr<ngraph::runtime::Allocator> m_allocator;
    size_t m_alignment;
};

// Abstarct class for the allocator, for allocating and deallocating device memory
class ngraph::runtime::Allocator
{
public:
    virtual void* cpu_malloc(void*, size_t size, size_t alignment) = 0;
    virtual void cpu_free(void* ptr, void*) = 0;
};

// SystemAllocator overides and implements "cpu_malloc" & "cpu_free" of Alloctaor interface class
// this class uses system library malloc and free for device memory allocation
class ngraph::runtime::SystemAllocator : public ngraph::runtime::Allocator
{
public:
    SystemAllocator(size_t alignment);
    ~SystemAllocator();

    void* cpu_malloc(void*, size_t size, size_t alignment) override
    {
        void* ptr = malloc(size);

        // check for exception
        if (size != 0 && !ptr)
        {
            throw ngraph_error("malloc failed to allocate memory of size " + std::to_string(size));
            throw std::bad_alloc();
        }
        return ptr;
    }

    void cpu_free(void* ptr, void*) override
    {
        if (ptr)
        {
            free(ptr);
        }
    }

private:
    size_t m_alignment;
};

// FrameworkAllocator overides and implements "cpu_malloc" & "cpu_free" of Alloctaor interface class,
// this class uses framework provide allocators and deallocators for device memory allocation
class ngraph::runtime::FrameworkAllocator : public ngraph::runtime::Allocator
{
public:
    FrameworkAllocator(AllocateFunc allocator, DestroyFunc deallocator, size_t alignment);
    ~FrameworkAllocator();

    void* cpu_malloc(void*, size_t size, size_t alignment) override
    {
        void* ptr = m_allocator(nullptr, alignment, size);

        // check for exception
        if (size != 0 && !ptr)
        {
            throw ngraph_error("malloc failed to allocate memory of size " + std::to_string(size));
            throw std::bad_alloc();
        }
        return ptr;
    }

    void cpu_free(void* ptr, void*) override
    {
        if (ptr)
        {
            m_deallocator(nullptr, ptr);
        }
    }

private:
    AllocateFunc m_allocator;
    DestroyFunc m_deallocator;
    size_t m_alignment;
};
