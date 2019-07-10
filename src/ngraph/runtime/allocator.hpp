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
#include <cstdlib>
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Allocator;
        class DefaultAllocator;
        /// \brief Create a default allocator that calls into system
        ///        allocation libraries
        ngraph::runtime::Allocator* get_default_allocator();
    }
}

/// \brief Abstract class for the allocator
class ngraph::runtime::Allocator
{
public:
    virtual ~Allocator() = 0;
    /// \brief allocates memory with the given size and alignment requirement
    /// \param size exact size of bytes to allocate
    /// \param alignment specifies the alignment. Must be a valid alignment supported by the implementation.
    virtual void* malloc(size_t size, size_t alignment) = 0;

    /// \brief deallocates the memory pointed by ptr
    /// \param ptr pointer to the aligned memory to be released
    virtual void free(void* ptr) = 0;
};
