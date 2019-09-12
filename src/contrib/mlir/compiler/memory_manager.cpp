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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#include "memory_manager.hpp"
#include <memory>
#include "ngraph/ngraph_visibility.hpp"

using namespace ngraph::runtime::ngmlir;

/// Call back to allocate memory for temps from JIT'ed code
extern "C" NGRAPH_API void* __mlir_allocate(MLIRMemMgr* memMgr, size_t size)
{
    return memMgr->allocate(size);
}

void* MLIRMemMgr::allocate(size_t size)
{
    void* ptr = malloc(size);
    ptrList.push_back(ptr);
    return ptr;
}

void MLIRMemMgr::freeAll()
{
    for (auto p : ptrList)
    {
        free(p);
    }
}
