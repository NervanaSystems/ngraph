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
#pragma once

#include "lowerer.hpp"
#include "memory_manager.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"

// TODO(dcab): Revisit and do fw decl when possible.
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/StandardOps/Ops.h>

namespace mlir
{
    struct StaticFloatMemRef;
}

namespace ngraph
{
    namespace op
    {
        class CompiledKernel;
    }
    namespace runtime
    {
        namespace ngmlir
        {
            class MLIRCompiler
            {
            public:
                static void init_mlir();

            public:
                using TensorList = std::vector<descriptor::Tensor*>;
                using TypeList = llvm::SmallVector<mlir::Type, 4>;

                MLIRCompiler(const ngraph::op::CompiledKernel* compiled_kernel,
                             const std::vector<void*>& external_tensors);

                /// Compiles and runs a subgraph in MLIR
                void compile_and_run();

                /// Returns the memory manager used by this sub-graph compiler
                MLIRMemMgr& get_mem_mgr() { return m_mem_mgr; }
                /// Returns memory manager pointer argument ID in call interface
                unsigned get_mem_mgr_arg_id(mlir::Function* func)
                {
                    return func->getNumArguments() - 1;
                }

            private:
                struct TensorInfo
                {
                    mlir::Value* m_value; /* mlir value this tensor maps to */
                    // More info here ?
                };

            private:
                void build_module();
                void lower_dialect();
                void optimize();
                void bind_arguments();
                void execute();
                void cleanup();

                mlir::Type get_mlir_type(const descriptor::Tensor* tensor);
                mlir::Type get_mlir_type(const element::Type& type);
                TensorInfo get_tensor_value(descriptor::Tensor* tensor);
                void update_tensor_value(descriptor::Tensor* tensor, mlir::Value* value);

                void build_ng_dialect();

                template <typename OP>
                static mlir::Value* create_op(MLIRCompiler& compiler, const ngraph::Node* ng_node)
                {
                    throw std::runtime_error("Unimplemented op '" + ng_node->description() +
                                             "' in MLIR Compiler");
                }

                template <typename BinOp>
                mlir::Value* create_binary_op(const ngraph::Node* ng_node);

                void create_return();

                /// Helper to create memref arguments for MLIR function signature
                llvm::SmallVector<void*, 8> allocate_memref_args(mlir::Function* func);

                /// Helper to allocate a mem ref object. Handles static shapes only for now.
                mlir::StaticFloatMemRef* allocate_memref_descriptor(mlir::Type type);

            private:
                mlir::MLIRContext m_context;
                std::unique_ptr<mlir::Module> m_module;
                std::unique_ptr<mlir::FuncBuilder> m_builder;
                std::unique_ptr<mlir::ExecutionEngine> m_engine;

                using TensorToInfo = std::pair<descriptor::Tensor*, TensorInfo>;
                using TensorToInfoMap = std::unordered_map<descriptor::Tensor*, TensorInfo>;
                using MLIRCompOpFunction =
                    std::function<mlir::Value*(MLIRCompiler& compiler, const ngraph::Node*)>;
                using MLIRCompOpMap = std::unordered_map<std::type_index, MLIRCompOpFunction>;

                // Sub-graph to be compiled and executed with MLIR.
                const ngraph::op::CompiledKernel* m_compiled_kernel;

                // Pointers to externally allocated memory for sub-graph's input and output tensors.
                const std::vector<void*>& m_external_tensors;
                llvm::SmallVector<void*, 8> m_invoke_args;

                // Maps tensor to the value it represents in the IR
                // use for MLIR dialect gen
                TensorToInfoMap m_tensor_to_value_map;
                static const MLIRCompOpMap op_dispatcher;

                // Memory manager for temp allocations inside JIT'ed code
                MLIRMemMgr m_mem_mgr;
            };
        }
    }
}
