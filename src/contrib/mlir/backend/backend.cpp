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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "ngraph/check.hpp"
#include "cpu_backend.hpp"
#include <memory>

using namespace ngraph::runtime::ngmlir;

/// Factory method to create new backends of certain kind. 
template<typename ...T>
static std::shared_ptr<MLIRBackend> MLIRBackend::create_backend(Kind kind, T&&... args)
{
    switch (kind)
    {
        case MLIRBackend::CPU:
            return std::make_shared<MLIRCPUBackend>(std::forward<T>(args)...);
        default:
            NGRAPH_UNREACHABLE("Unsupported MLIR backend");
    }
}
