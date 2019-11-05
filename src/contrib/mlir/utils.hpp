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

#include <mlir/IR/Module.h>

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            /// Common nGraph dialect initialization code. Used by nGraph compiler and tools that
            /// require nGraph dialect initialization.
            void initializeNGraphMLIR();

            /// Helper to dump MLIR module into llvm::dbgs prepended by the message \p msg.
            void dumpMlirModule(const std::string msg, mlir::ModuleOp module);
        } // namespace ngmlir
    }     // namespace runtime
} // namespace ngraph
