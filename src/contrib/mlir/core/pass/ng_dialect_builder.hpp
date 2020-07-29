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
// Follows nGraph naming convention for public APIs only, else MLIR naming
// convention.

#pragma once

#include "contrib/mlir/core/compiler.hpp"

#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"

#include <mlir/Pass/Pass.h>

namespace ngraph
{
    namespace pass
    {
        std::unique_ptr<mlir::Pass>
            createNgDialectConversionPass(std::shared_ptr<ngraph::Function> function,
                                          mlir::MLIRContext* context);

        /// NgDialectConversionPass is an MLIR Pass Given an nGraph sub-graph,
        /// represented as an ngraph::Function, it
        /// translates the graph down to nGraph dialect

        class NgDialectConversionPass
            : public mlir::PassWrapper<NgDialectConversionPass, mlir::OperationPass<mlir::ModuleOp>>
        {
        public:
            using TensorList = std::vector<descriptor::Tensor*>;
            using TypeList = llvm::SmallVector<mlir::Type, 4>;

            NgDialectConversionPass(std::shared_ptr<ngraph::Function> function,
                                    mlir::MLIRContext* context)
                : m_function(function)
                , m_context(context)
                , m_builder(context)
            {
            }

            NgDialectConversionPass(const NgDialectConversionPass& obj);

        private:
            struct TensorInfo
            {
                // MLIR values this tensor maps to.
                mlir::Value m_value;
            };

            // Converts an nGraph sub-graph to MLIR nGraph dialect.
            void buildNgDialectModule();
            void buildNgDialect(mlir::FuncOp function);
            void runOnOperation() override;

            mlir::Type getMlirType(const descriptor::Tensor* tensor);
            mlir::Type getMlirType(const element::Type& type);

            TensorInfo getTensorValue(descriptor::Tensor* tensor);
            void updateTensorValue(descriptor::Tensor* tensor, mlir::Value value);

            template <typename Op>
            static mlir::Operation* createOp(NgDialectConversionPass& NgDialectObj,
                                             const ngraph::Node* ngNode)
            {
                throw std::runtime_error("Unimplemented op '" + ngNode->description() +
                                         "' in MLIR Compiler");
            }

            // Generic op lowerer to ng dialect.
            // Simply maps ngraph tensors to values and generate an OP. No op-specific
            // logic.
            // Use inNum when mlir OP needs less input than its corresponding ngraph OP.
            template <typename Op>
            mlir::Operation* createGenericOp(const ngraph::Node* ngNode, int inNum = -1)
            {
                std::vector<mlir::Value> argValues;
                std::vector<mlir::Type> resTypes;
                std::shared_ptr<descriptor::Tensor> argTensor;
                int i = 0;
                for (auto& argOutput : ngNode->input_values())
                {
                    if (inNum != -1 && i == inNum)
                    {
                        break;
                    }
                    argTensor = argOutput.get_tensor_ptr();
                    auto argV = getTensorValue(argTensor.get()).m_value;
                    argValues.push_back(argV);
                    i++;
                }

                for (auto& output : ngNode->outputs())
                {
                    resTypes.push_back(getMlirType(output.get_tensor_ptr().get()));
                }

                return (m_builder.create<Op,
                                         llvm::ArrayRef<mlir::Type>,
                                         llvm::ArrayRef<mlir::Value>,
                                         llvm::ArrayRef<mlir::NamedAttribute>>(
                            mlir::UnknownLoc::get(m_context),
                            resTypes,
                            argValues,
                            {/* no attrs */}))
                    .getOperation();
            }

            template <typename RedOp>
            mlir::Operation* createIndexReduction(const ngraph::Node* ngNode)
            {
                auto* idxRed = static_cast<const ngraph::op::util::IndexReduction*>(ngNode);
                auto op = createGenericOp<RedOp>(ngNode);
                mlir::ArrayAttr redAxesAttr =
                    m_builder.getI64ArrayAttr({(int64_t)idxRed->get_reduction_axis()});
                op->setAttr("axes", redAxesAttr);
                return op;
            }

            void createReturn();

            /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
            template <typename T>
            void getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape)
            {
                for (auto dim : ngShape)
                {
                    mlirShape.push_back(dim);
                }
            }

            /// Converts an ngraph shape to an I64 array attribute
            template <typename T>
            mlir::ArrayAttr getShapeAsAttr(T ngShape)
            {
                llvm::SmallVector<int64_t, 4> mlirShape;
                getMlirShape(ngShape, mlirShape);
                return m_builder.getI64ArrayAttr(mlirShape);
            }

            /// Returns the builder
            mlir::OpBuilder& getBuilder() { return m_builder; }
            /// Return the real input node corresponding to the fake node
            ngraph::Node* getOriginArg(ngraph::Node* node) const;

            // Sub-graph to be compiled and executed with MLIR.
            std::shared_ptr<Function> m_function;

            // MLIR context that holds all the MLIR information related to the sub-graph
            // compilation.
            mlir::MLIRContext* m_context;
            mlir::OpBuilder m_builder;

            using TensorToInfo = std::pair<descriptor::Tensor*, TensorInfo>;
            using TensorToInfoMap = std::unordered_map<descriptor::Tensor*, TensorInfo>;
            using MLIRCompOpFunction = std::function<mlir::Operation*(
                NgDialectConversionPass& NgDialectObj, const ngraph::Node*)>;
            using MLIRCompOpMap = std::unordered_map<Node::type_info_t, MLIRCompOpFunction>;

            // Maps tensor to the value it represents in the IR
            // use for MLIR dialect gen
            TensorToInfoMap m_tensorToValueMap;
            static const MLIRCompOpMap& getOpDispatcher();
        };
    }
}
