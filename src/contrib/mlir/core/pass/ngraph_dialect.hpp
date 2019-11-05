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

#include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/runtime/cpu/memory_manager.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"

#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>

#include <typeindex>
#include <unordered_map>
#include <vector>

using namespace ngraph::runtime::ngmlir;

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
    namespace pass
    {
        /// MLIR Compiler. Given an nGraph sub-graph, represented as CompiledKernel node, it
        /// translates the graph down to nGraph dialect and applies core optimizations.
        ///
        /// The compiler owns the MLIR module until compilation is done. After that,
        /// the module can be grabbed and plugged into MLIR backends.
        class NgDialectConversionPass : public mlir::ModulePass<NgDialectConversionPass>
        {
        public:
            using TensorList = std::vector<descriptor::Tensor*>;
            using TypeList = llvm::SmallVector<mlir::Type, 4>;

            NgDialectConversionPass(const ngraph::op::CompiledKernel* compiled_kernel,
                                    mlir::MLIRContext* context,
                                    mlir::OpBuilder* builder,
                                    MLIRCompiler* compiler)
                : m_compiledKernel(compiled_kernel)
                , m_context(context)
                , m_builder(builder)
                , m_compiler(compiler)
            {
            }

            NgDialectConversionPass(const NgDialectConversionPass& obj);

        private:
            struct TensorInfo
            {
                // MLIR values this tensor maps to.
                mlir::Value* m_value;
            };

        private:
            // Converts an nGraph sub-graph to MLIR nGraph dialect.
            void buildNgDialectModule();
            void buildNgDialect();
            void runOnModule() override;
            // Applies any nGraph dialect optimizations
            void optimizeNgDialect() { /*TODO: Add Core NG dialect optimizations */}

            mlir::Type getMlirType(const descriptor::Tensor* tensor);
            mlir::Type getMlirType(const element::Type& type);
            mlir::Type getMlirType(const ngraph::Node* node);

            TensorInfo getTensorValue(descriptor::Tensor* tensor);
            void updateTensorValue(descriptor::Tensor* tensor, mlir::Value* value);

            template <typename Op>
            static mlir::Operation* createOp(NgDialectConversionPass& NgDialectObj,
                                             const ngraph::Node* ngNode)
            {
                throw std::runtime_error("Unimplemented op '" + ngNode->description() +
                                         "' in MLIR Compiler");
            }

            // Generic op lowerer to ng dialect.
            // Simply maps ngraph tensors to values and generate an OP. No op-specific logic.
            template <typename Op>
            mlir::Operation* createGenericOp(const ngraph::Node* ngNode);

            template <typename RedOp>
            mlir::Operation* createIndexReduction(const ngraph::Node* ngNode);

            void createReturn();

            /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
            template <typename T>
            void getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape);

            /// Converts an ngraph shape to an I64 array attribute
            template <typename T>
            mlir::ArrayAttr getShapeAsAttr(T ngShape);

        private:
            // Sub-graph to be compiled and executed with MLIR.
            const ngraph::op::CompiledKernel* m_compiledKernel;
            MLIRCompiler* m_compiler;

            // MLIR context that holds all the MLIR information related to the sub-graph
            // compilation.
            mlir::MLIRContext* m_context;
            mlir::OpBuilder* m_builder;

            using TensorToInfo = std::pair<descriptor::Tensor*, TensorInfo>;
            using TensorToInfoMap = std::unordered_map<descriptor::Tensor*, TensorInfo>;
            using MLIRCompOpFunction = std::function<mlir::Operation*(
                NgDialectConversionPass& NgDialectObj, const ngraph::Node*)>;
            using MLIRCompOpMap = std::unordered_map<std::type_index, MLIRCompOpFunction>;

            // Maps tensor to the value it represents in the IR
            // use for MLIR dialect gen
            TensorToInfoMap m_tensorToValueMap;
            static const MLIRCompOpMap opDispatcher;
        };
        std::unique_ptr<mlir::Pass>
            CreateNgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel,
                                          mlir::MLIRContext* context,
                                          mlir::OpBuilder* builder,
                                          MLIRCompiler* compiler);
    }
}
