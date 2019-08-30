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

// NOTE: This file follows nGraph format style and naming convention since it
// exposes a public API to the rest of nGraph codebase.

#pragma once

#include "memory_manager.hpp"
#include "ngraph/node.hpp"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>

#include <typeindex>
#include <unordered_map>
#include <vector>

namespace llvm
{
    class TargetMachine;
}

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
            /// This class is the entry point to MLIR from nGraph. It drives the conversion of
            /// nGraph sub-graphs, represented with CompiledKernel nodes, to MLIR nGraph dialect
            /// and its lowering, optimization and execution using LLVM-based MLIR execution engine.
            class MLIRCompiler
            {
            public:
                /// Initializes MLIR environment. It must be called only once per execution.
                static void init_mlir();

            public:
                using TensorList = std::vector<descriptor::Tensor*>;
                using TypeList = llvm::SmallVector<mlir::Type, 4>;

                MLIRCompiler(const ngraph::op::CompiledKernel* compiled_kernel)
                    : m_compiled_kernel(compiled_kernel)
                {
                }

                /// Compiles a subgraph with MLIR
                void compile();

                /// Executes a pre-compiled subgraph
                void run(std::vector<void*>& external_tensors);

                /// Returns the memory manager used by this sub-graph compiler.
                MLIRMemMgr& get_mem_mgr() { return m_mem_mgr; }
                /// Returns memory manager pointer argument ID in call interface.
                unsigned get_mem_mgr_arg_id(mlir::FuncOp& func);

            private:
                struct TensorInfo
                {
                    // MLIR values this tensor maps to.
                    mlir::Value* m_value;
                };

            private:
                void build_ng_dialect_module();
                void lower_ng_dialect();
                void optimize();
                void bind_arguments(std::vector<void*>& external_tensors);
                void execute();
                void cleanup();

                mlir::Type get_mlir_type(const descriptor::Tensor* tensor);
                mlir::Type get_mlir_type(const element::Type& type);
                mlir::Type get_mlir_type(const ngraph::Node* node);

                TensorInfo get_tensor_value(descriptor::Tensor* tensor);
                void update_tensor_value(descriptor::Tensor* tensor, mlir::Value* value);

                void build_ng_dialect();

                template <typename Op>
                static mlir::Operation* create_op(MLIRCompiler& compiler,
                                                  const ngraph::Node* ng_node)
                {
                    throw std::runtime_error("Unimplemented op '" + ng_node->description() +
                                             "' in MLIR Compiler");
                }

                // Generic op lowerer to ng dialect.
                // Simply maps ngraph tensors to values and generate an OP. No op-specific logic.
                template <typename Op>
                mlir::Operation* create_generic_op(const ngraph::Node* ng_node);

                template <typename RedOp>
                mlir::Operation* create_index_reduction(const ngraph::Node* ng_node);

                void create_return();

                /// Helper to create memref arguments for MLIR function signature
                llvm::SmallVector<void*, 8> allocate_memref_args();

                /// Helper to allocate a mem ref object. Handles static shapes only for now.
                mlir::StaticFloatMemRef* allocate_memref_descriptor();

                /// Helper to dump MLIR module into llvm::dbgs prepended by the message \p msg.
                void dump_mlir_module(const std::string msg);

                /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
                template <typename T>
                void get_mlir_shape(T ng_shape, llvm::SmallVectorImpl<int64_t>& mlir_shape);

                /// Converts an ngraph shape to an I64 array attribute
                template <typename T>
                mlir::ArrayAttr get_shape_as_attr(T ng_shape);

            private:
                // Sub-graph to be compiled and executed with MLIR.
                const ngraph::op::CompiledKernel* m_compiled_kernel;

                // Pointers to externally allocated memory for sub-graph's input and output tensors.
                std::vector<void*>* m_external_tensors;

                // Arguments for the MLIR function generated for the nGraph sub-graph.
                llvm::SmallVector<void*, 8> m_invoke_args;

                // MLIR context that holds all the MLIR information related to the sub-graph
                // compilation.
                mlir::MLIRContext m_context;

                mlir::OwningModuleRef m_module;
                std::unique_ptr<mlir::OpBuilder> m_builder;
                std::unique_ptr<mlir::ExecutionEngine> m_engine;

                using TensorToInfo = std::pair<descriptor::Tensor*, TensorInfo>;
                using TensorToInfoMap = std::unordered_map<descriptor::Tensor*, TensorInfo>;
                using MLIRCompOpFunction =
                    std::function<mlir::Operation*(MLIRCompiler& compiler, const ngraph::Node*)>;
                using MLIRCompOpMap = std::unordered_map<std::type_index, MLIRCompOpFunction>;

                // Maps tensor to the value it represents in the IR
                // use for MLIR dialect gen
                TensorToInfoMap m_tensor_to_value_map;
                static const MLIRCompOpMap op_dispatcher;

                // Memory manager for temp allocations inside JIT'ed code
                MLIRMemMgr m_mem_mgr;

                // Optimization level used by MLIR and LLVM compilers.
                static unsigned mlir_opt_level;

                // LLVM target machine to be used by this MLIR compiler instance to retrieve
                // information about target features.
                // TODO: Note that, unfortunatelly, MLIR/OrcJIT execution engine creates its own
                // target machine for compilation internally. This target machine is for non-JIT
                // related stuff. We should change OrcJIT API so that we can pass an external target
                // machine or configuration flags.
                // TODO: Move target machine to external nGraph backend when multiple backends start
                // to use MLIR.
                static std::unique_ptr<llvm::TargetMachine> target_machine;
            };
        }
    }
}
