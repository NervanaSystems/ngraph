//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "contrib/mlir/backend/backend.hpp"
#include "llvm/Support/CodeGen.h"
#include "ngraph/check.hpp"

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
            class MLIRCPUBackend : public MLIRBackend
            {
            public:
                /// Global Initialization for all CPU backends
                static void init();

                MLIRCPUBackend(mlir::OwningModuleRef& module, mlir::MLIRContext& context)
                    : MLIRBackend(module, context)
                {
                    NGRAPH_CHECK(initialized,
                                 "Cannot instantiate CPU MLIR backend without initialization");
                }

                MLIRCPUBackend(mlir::ModuleOp& moduleOp, mlir::MLIRContext& context)
                    : MLIRBackend(moduleOp, context)
                {
                    NGRAPH_CHECK(initialized,
                                 "Cannot instantiate CPU MLIR backend without initialization");
                }
                // codegen LLVM dialect from nGraph dialect applying CPU backend optimization passes
                void codegen() override;

            private:
                // Apply CPU specific optimizations at nGraph dialect level
                void optimizeNgDialect();

                // Lowers nGraph dialect all the way to Affine dialect.
                void lowerNgDialect();

                // Lowers standard dialect all the way to LLVM dialect.
                void lowerStandardDialect();

                // Apply affine dialect optimizations
                void optimizeAffineDialect();

            public:
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
                // Global initialization done for CPU backend
                static bool initialized;
            };
        }
    }
}
