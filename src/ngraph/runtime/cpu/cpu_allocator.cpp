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

#include "ngraph/runtime/cpu/cpu_allocator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            CPUAllocator::CPUAllocator() {}
            CPUAllocator::~CPUAllocator() {}
            CPUAllocator& GetCPUAllocator()
            {
                static CPUAllocator cpu_allocator;
                return cpu_allocator;
            }
        }
    }
}
/*ngraph::runtime::SystemAllocator::SystemAllocator()
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
}*/
