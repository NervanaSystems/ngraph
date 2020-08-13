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

#include "Conversion/NgraphToStandard/NgraphToStandard.hpp"
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Affine/EDSC/Builders.h>
#include <mlir/Dialect/Affine/EDSC/Intrinsics.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/EDSC/Intrinsics.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::edsc::op;

namespace
{
    // class LowerNgraphPass : public ConvertNgraphToStandardBase<LowerNgraphPass>
    // {
    //     void runOnOperation() override
    //     {
    //         OwningRewritePatternList patterns;
    //         populateNgraphToStdConversionPatterns(patterns, &getContext());
    //         populateNgraphToVectorConversionPatterns(patterns, &getContext());
    //         ConversionTarget target(getContext());
    //         target.addLegalDialect<scf::SCFDialect, StandardOpsDialect, VectorDialect>();
    //         if (failed(applyPartialConversion(getOperation(), target, patterns)))
    //             signalPassFailure();
    //     }
    // };

    // /// Base class for nGraph operation conversions to affine/standard dialect.
    // /// Provides conversion patterns with an access to the DialectLoweringPass
    // /// which holds the state of the conversion.
    // class NGraphOpConversionBase : public ::mlir::ConversionPattern
    // {
    // public:
    //     NGraphOpConversion(llvm::StringRef rootOpName, ::mlir::MLIRContext* context,
    //     DialectLoweringPass& pass)
    //         : ConversionPattern(rootOpName, /*benefit=*/1, context)
    //         , pass(pass)
    //         , context(context){};

    // protected:
    //     // Back-reference to the lowering pass which contains the lowering state,
    //     // including the nGraph type converter.
    //     DialectLoweringPass& pass;
    //     MLIRContext* context;
    // };

    // class AddOpConversion : public NGraphOpLowering
    // {
    // public:
    //     explicit AddOpConversion(mlir::MLIRContext* context, DialectLoweringPass& pass)
    //         : NGraphOpLowering(mlir::Ngraph_AddOp::getOperationName(), context, pass)
    //     {
    //     }

    //     LogicalResult matchAndRewrite(Operation* op,
    //                                   ArrayRef<Value> operands,
    //                                   ConversionPatternRewriter& rewriter) const override
    //     {
    //     }
    // };
}
