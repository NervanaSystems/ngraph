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

#pragma once

#include <memory>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>
#include "contrib/mlir/backend/backend.hpp"
#include "contrib/mlir/runtime/runtime.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            /// A CPU Runtime is an MLIR runtime that owns an MLIR context and a module
            /// The module should be in LLVM dialect and ready to be lowered via an MLIR
            /// ExecutionEngine. The runtime owns the context and must out-live any MLIR
            /// code Compilation and execution.
            class MLIRCPURuntime : public MLIRRuntime
            {
            public:
                /// Executes a pre-compiled subgraph
                void run(void* args) override;

            private:
                void run_internal(std::vector<void*>& externalTensors);
                // Bind external tensors to MLIR module entry point
                void bindArguments(std::vector<void*>& externalTensors);
                // Invokes an MLIR module entry point with bound arguments
                void execute();
                // Cleans up allocated args
                void cleanup();

                /// Helper to create memref arguments for MLIR function signature
                llvm::SmallVector<void*, 8> allocateMemrefArgs();

                /// Helper to allocate a mem ref object. Handles static shapes only for now.
                mlir::StaticFloatMemRef* allocateMemrefDescriptor();

            private:
                // Pointers to externally allocated memory for sub-graph's input and output tensors.
                std::vector<void*>* m_externalTensors;
                // Arguments for the MLIR function generated for the nGraph sub-graph.
                llvm::SmallVector<void*, 8> m_invokeArgs;
                std::unique_ptr<mlir::ExecutionEngine> m_engine;
            };
        }
    }
}
