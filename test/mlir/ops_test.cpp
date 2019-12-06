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

// ops tests for nGraph MLIR dialect
// Test certain invariants about
#include "gtest/gtest.h"

#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"
#include "contrib/mlir/utils.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

OpBuilder createBuilder(MLIRContext* context)
{
    auto module = ModuleOp::create(UnknownLoc::get(context));
    auto funcType = FunctionType::get({}, {}, context);
    auto function = FuncOp::create(UnknownLoc::get(context), "main", funcType);
    function.addEntryBlock();

    OpBuilder builder(function.getBody());
    return builder;
}

TEST(MLIR, op_version_interface)
{
    MLIRContext context;
    llvm::SmallVector<mlir::Type, 1> resultTypes;

    OpBuilder builder(&context);
    resultTypes.push_back(
        mlir::NGTensorType::get(&context, mlir::NGFloatType::getF16(&context), {2, 2}));

    auto operation = Operation::create(mlir::UnknownLoc::get(&context),
                                       OperationName("ng.gather", &context),
                                       resultTypes,
                                       llvm::None,
                                       llvm::None,
                                       llvm::None,
                                       0,
                                       false);

    EXPECT_TRUE(llvm::dyn_cast<OpVersion0>(operation) != nullptr);
    EXPECT_TRUE(llvm::dyn_cast<OpVersion1>(operation) == nullptr);
}

TEST(MLIR, fused_ops_interface)
{
    MLIRContext context;

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    OpBuilder builder(&context);
    resultTypes.push_back(
        mlir::NGTensorType::get(&context, mlir::NGFloatType::getF16(&context), {2, 2}));

    auto operation = Operation::create(mlir::UnknownLoc::get(&context),
                                       OperationName("ng.squeeze", &context),
                                       resultTypes,
                                       llvm::None,
                                       llvm::None,
                                       llvm::None,
                                       0,
                                       false);

    EXPECT_TRUE(llvm::dyn_cast<FusedOp>(operation) != nullptr);
    if (auto fusedOp = llvm::dyn_cast<FusedOp>(operation))
    {
        fusedOp.decompose();
    }
}

TEST(MLIR, ops_attributes)
{
    MLIRContext context;
    auto resultType =
        mlir::NGTensorType::get(&context, mlir::NGFloatType::getF16(&context), {2, 2});
    auto builder = createBuilder(&context);

    auto def = builder.create<NGConstantOp>(UnknownLoc::get(&context),
                                            resultType,
                                            builder.getI64ArrayAttr({2, 3, 4}),
                                            builder.getF32ArrayAttr({1.0, 2.3, 5.6}));
    auto operation =
        builder
            .create<NGAvgPoolOp>(
                UnknownLoc::get(&context),
                resultType,
                def.getResult(),                    // arg
                builder.getI64ArrayAttr({2, 3, 4}), // windowShape
                builder.getI64ArrayAttr({2, 3, 4}), // windowMovementStrides
                builder.getI64ArrayAttr({0, 0, 0}), // padBelow
                builder.getI64ArrayAttr({0, 0, 0}), // padAbove
                builder.getBoolAttr(false),         // includePadding
                builder.getI64IntegerAttr(static_cast<int64_t>(MLIRPadType::SAME_LOWER)), // padType
                builder.getBoolAttr(false)) // ceilMode
            .getOperation();

    auto avgPool = cast<NGAvgPoolOp>(operation);
    auto padType = avgPool.padType();
    EXPECT_TRUE(padType == MLIRPadType::SAME_LOWER);

    operation =
        builder
            .create<NGAvgPoolOp>(UnknownLoc::get(&context),
                                 resultType,
                                 def.getResult(),                    // arg
                                 builder.getI64ArrayAttr({2, 3, 4}), // windowShape
                                 builder.getI64ArrayAttr({2, 3, 4}), // windowMovementStrides
                                 builder.getI64ArrayAttr({0, 0, 0}), // padBelow
                                 builder.getI64ArrayAttr({0, 0, 0}), // padAbove
                                 builder.getBoolAttr(false))         // includePadding
            .getOperation();

    avgPool = cast<NGAvgPoolOp>(operation);
    padType = avgPool.padType();
    EXPECT_TRUE(padType == MLIRPadType::EXPLICIT);

    auto ceilMode = avgPool.ceilMode();
    EXPECT_TRUE(ceilMode == false);
}
