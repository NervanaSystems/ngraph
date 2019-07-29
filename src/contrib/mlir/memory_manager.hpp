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

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <vector>

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            /// Memory manager for temporaries in MLIR compiled sub-graph
            /// It handles call-backs from the code and returns pointer to allocated memory
            /// Also, handles freeing up memory
            class MLIRMemMgr
            {
            public:
                /// Allocates data for temporary tensor. Currently, it is called for each
                /// temp tensor defintion. Keeps track of each pointer and free them during cleanup.
                // TODO: Use pre-allocation from framework memory manager
                void* allocate(size_t size);

                /// Frees all allocated pointers
                void freeAll();

            private:
                std::vector<void*> ptrList;
            };
        }
    }
}
