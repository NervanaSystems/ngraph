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

#include "contrib/mlir/compiler/dialect/dialect.hpp"
#include "contrib/mlir/compiler/dialect/ops.hpp"
#include "contrib/mlir/compiler/dialect/type.hpp"
#include "contrib/mlir/compiler/tools.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

void initNgDialect()
{
    static bool init = false;
    if (!init)
    {
        ngraph::runtime::ngmlir::initializeNGraphMLIR();
        init = true;
    }
}
TEST(MLIR, op_version_interface)
{
    // Initialize before any context declarations
    initNgDialect();
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
    // Initialize before any context declarations
    initNgDialect();
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
