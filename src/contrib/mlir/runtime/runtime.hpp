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

#include "contrib/mlir/backend/backend.hpp"

#include <memory>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            /// Base class for an MLIR runtime. An MLIR runtime owns the MLIR Context and owns
            /// the final compiled module. It supports invoking the module with specific arguments
            class MLIRRuntime
            {
            public:
                /// Sets the MLIR module that this runtime will own
                void set_module(mlir::OwningModuleRef& module) { m_module = std::move(module); }
                /// Overload with module op
                void set_module(mlir::ModuleOp& module) { m_module = module; }
                /// Executes a pre-compiled subgraph
                virtual void run(void* args) = 0;

                /// Get the MLIR module that this runtime owns
                mlir::OwningModuleRef& get_module() { return m_module; }
                mlir::MLIRContext& get_context() { return m_context; }
            protected:
                mlir::OwningModuleRef m_module;
                mlir::MLIRContext m_context;
            };
        }
    }
}