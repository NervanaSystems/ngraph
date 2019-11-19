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

#include "contrib/mlir/runtime/cpu/memory_manager.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>

#include <typeindex>
#include <unordered_map>
#include <vector>

namespace ngraph
{
    namespace descriptor
    {
        class Tensor;
    }
    namespace element
    {
        class Type;
    }
    namespace op
    {
        class CompiledKernel;
    }
    namespace runtime
    {
        namespace ngmlir
        {
            /// MLIR Compiler. Given an nGraph sub-graph, represented as CompiledKernel node, it
            /// translates the graph down to nGraph dialect and applies core optimizations.
            ///
            /// The compiler owns the MLIR module until compilation is done. After that,
            /// the module can be grabbed and plugged into MLIR backends.
            class MLIRCompiler
            {
            public:
                /// Initializes MLIR environment. It must be called only once.
                static void init();

            public:
                MLIRCompiler(const ngraph::op::CompiledKernel* compiled_kernel,
                             mlir::MLIRContext& context)
                    : m_compiledKernel(compiled_kernel)
                    , m_context(context)
                {
                    NGRAPH_CHECK(initialized,
                                 "Cannot instantiate a compiler without initializing MLIR");
                }

                /// Compiles a subgraph with MLIR
                void compile();

                mlir::OwningModuleRef& get_module() { return m_module; }
            private:
                // Converts an nGraph sub-graph to MLIR nGraph dialect.
                void buildNgDialectModule();
                // Applies any nGraph dialect optimizations
                void optimizeNgDialect() { /*TODO: Add Core NG dialect optimizations */}

            private:
                // Sub-graph to be compiled and executed with MLIR.
                const ngraph::op::CompiledKernel* m_compiledKernel;

                // MLIR context that holds all the MLIR information related to the sub-graph
                // compilation.
                mlir::MLIRContext& m_context;
                mlir::OwningModuleRef m_module;

                // Global initialization for MLIR compiler
                static bool initialized;
            };
        }
    }
}
