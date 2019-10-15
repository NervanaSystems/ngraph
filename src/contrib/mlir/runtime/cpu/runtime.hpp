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

#include "contrib/mlir/backend/backend.hpp"

#include <memory>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>

namespace llvm
{
    class TargetMachine;
}


namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            class MLIRCPURuntime
            {
                public:
                MLIRCPURuntime(mlir::ModuleOp module) 
                : m_module(module) {}
                /// Executes a pre-compiled subgraph
                void run(std::vector<void*>& externalTensors);
                
                mlir::OwningModuleRef& get_module()
                {
                    return m_module;
                }
                private:
                void bindArguments(std::vector<void*>& externalTensors);
                void execute();
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
                mlir::OwningModuleRef m_module;
                std::unique_ptr<mlir::ExecutionEngine> m_engine;

                // JIT optimization level
                static llvm::CodeGenOpt::Level mlirOptLevel;

                // LLVM target machine to be used by this MLIR compiler instance to retrieve
                // information about target features.
                // TODO: Note that, unfortunatelly, MLIR/OrcJIT execution engine creates its own
                // target machine for compilation internally. This target machine is for non-JIT
                // related stuff. We should change OrcJIT API so that we can pass an external target
                // machine or configuration flags.
                // TODO: Move target machine to external nGraph backend when multiple backends start
                // to use MLIR.
                static std::unique_ptr<llvm::TargetMachine> targetMachine;
            };
        }
    }
}