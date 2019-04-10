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

#include "ngraph/runtime/allocator.hpp"

ngraph::runtime::Allocator::~Allocator()
{
}

class DefaultNgraphAllocator : public ngraph::runtime::Allocator
{
public:
    void* Malloc(size_t size, size_t alignment)
    {
        void* ptr = ngraph::aligned_alloc(alignment, size);

        // check for exception
        if (!ptr)
        {
            throw ngraph::ngraph_error("malloc failed to allocate memory of size " +
                                       std::to_string(size));
        }
        return ptr;
    }

    void Free(void* ptr)
    {
        if (ptr)
        {
            ngraph::aligned_free(ptr);
        }
    }
};

ngraph::runtime::Allocator* ngraph::runtime::get_ngraph_allocator()
{
    static std::unique_ptr<DefaultNgraphAllocator> allocator(new DefaultNgraphAllocator());
    return allocator.get();
}
