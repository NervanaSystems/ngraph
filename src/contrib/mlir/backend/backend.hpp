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
#include <mlir/IR/Module.h>

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            class MLIRBackend
            {
            public:
                MLIRBackend(mlir::OwningModuleRef& module, mlir::MLIRContext& context)
                    : m_module(std::move(module))
                    , m_context(context)
                {
                }

                MLIRBackend(mlir::ModuleOp& moduleOp, mlir::MLIRContext& context)
                    : m_module(moduleOp)
                    , m_context(context)
                {
                }

                /// Generate code for the module
                virtual void codegen() = 0;

                mlir::OwningModuleRef& get_module() { return m_module; }
            protected:
                mlir::OwningModuleRef m_module;
                mlir::MLIRContext& m_context;
            };
        }
    }
}
