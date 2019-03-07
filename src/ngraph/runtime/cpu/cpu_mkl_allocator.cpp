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
#include "ngraph/runtime/cpu/cpu_external_function.hpp"

ngraph::runtime::cpu::CPUAllocator::CPUAllocator(ngraph::runtime::Allocator* allocator,
                                                 size_t alignment)
    : m_allocator(std::move(allocator))
    , m_alignment(alignment)
{
}

ngraph::runtime::cpu::CPUAllocator::~CPUAllocator()
{
}

void* ngraph::runtime::cpu::CPUAllocator::malloc(size_t size)
{
    return m_allocator->cpu_malloc(nullptr, size, m_alignment);
}

void ngraph::runtime::cpu::CPUAllocator::free(void* ptr)
{
    m_allocator->cpu_free(nullptr, ptr);
}

ngraph::runtime::SystemAllocator::SystemAllocator()
{
}

ngraph::runtime::SystemAllocator::~SystemAllocator()
{
}

ngraph::runtime::FrameworkAllocator::~FrameworkAllocator()
{
}

ngraph::runtime::FrameworkAllocator::FrameworkAllocator(AllocateFunc& allocator,
                                                        DestroyFunc& deallocator)
    : m_allocator(allocator)
    , m_deallocator(deallocator)
{
}
