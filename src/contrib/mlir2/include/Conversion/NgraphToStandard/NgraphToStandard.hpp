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

#pragma once

#include "mlir/Support/LLVM.h"

namespace mlir
{
    class AffineExpr;
    class AffineForOp;
    class AffineMap;
    class AffineParallelOp;
    class Location;
    struct LogicalResult;
    class MLIRContext;
    class OpBuilder;
    class Pass;
    class RewritePattern;
    class Value;
    class ValueRange;

    // Owning list of rewriting patterns.
    class OwningRewritePatternList;

    // /// Emit code that computes the given affine expression using standard
    // /// arithmetic operations applied to the provided dimension and symbol values.
    // Value expandAffineExpr(OpBuilder& builder,
    //                        Location loc,
    //                        AffineExpr expr,
    //                        ValueRange dimValues,
    //                        ValueRange symbolValues);

    // /// Create a sequence of operations that implement the `affineMap` applied to
    // /// the given `operands` (as it it were an AffineApplyOp).
    // Optional<SmallVector<Value, 8>>
    //     expandAffineMap(OpBuilder& builder, Location loc, AffineMap affineMap, ValueRange operands);

    // /// Collect a set of patterns to convert from the Affine dialect to the Standard
    // /// dialect, in particular convert structured affine control flow into CFG
    // /// branch-based control flow.
    // void populateAffineToStdConversionPatterns(OwningRewritePatternList& patterns,
    //                                            MLIRContext* ctx);

    // /// Collect a set of patterns to convert vector-related Affine ops to the Vector
    // /// dialect.
    // void populateAffineToVectorConversionPatterns(OwningRewritePatternList& patterns,
    //                                               MLIRContext* ctx);

    // /// Emit code that computes the lower bound of the given affine loop using
    // /// standard arithmetic operations.
    // Value lowerAffineLowerBound(AffineForOp op, OpBuilder& builder);

    // /// Emit code that computes the upper bound of the given affine loop using
    // /// standard arithmetic operations.
    // Value lowerAffineUpperBound(AffineForOp op, OpBuilder& builder);

    /// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
    /// to equivalent lower-level constructs (flow of basic blocks and arithmetic
    /// primitives).
    std::unique_ptr<Pass> createLowerAffinePass();
}
