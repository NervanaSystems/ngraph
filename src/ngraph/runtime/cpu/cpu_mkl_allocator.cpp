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

#include "ngraph/runtime/cpu/cpu_mkl_allocator.hpp"
#include <string>
#include "ngraph/except.hpp"

ngraph::runtime::cpu::CPUAllocator::CPUAllocator()
    : m_framework_allocator(nullptr)
    , m_framework_deallocator(nullptr)
    , m_alignment(0)
{
}

ngraph::runtime::cpu::CPUAllocator::CPUAllocator(AllocateFunc allocator,
                                                 DestroyFunc deallocator,
                                                 size_t alignment)
    : m_framework_allocator(allocator)
    , m_framework_deallocator(deallocator)
    , m_alignment(alignment)
{
    //    mkl::i_malloc = MallocHook;
    //    mkl::i_free = FreeHook;
}

void* ngraph::runtime::cpu::CPUAllocator::cpu_malloc(size_t size)
{
    void* ptr;
    if (m_framework_allocator != nullptr)
    {
        ptr = m_framework_allocator(nullptr, m_alignment, size);
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

void ngraph::runtime::cpu::CPUAllocator::cpu_free(void* ptr)
{
    if (m_framework_deallocator && ptr)
    {
        m_framework_deallocator(nullptr, ptr);
    }
    else if (ptr)
    {
        free(ptr);
    }
}

ngraph::runtime::cpu::CPUAllocator::~CPUAllocator()
{
}
