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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#include "ops.hpp"
#include "assertion.hpp"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "type.hpp"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;
using namespace mlir;
// TODO:
// - Move verifiers and other OP helpers (e.g. getSomeAttribute()) to separate files
//
// - Op helpers: Since it is not possible to add arbitrary code (and would complicate the .td file)
// to Ops classes, we will add helper classes with static methods for each Op that needs it
// Additional verification methods
// Tensor type checks are already verified by the caller of these methods

/// Checks if all operands and results are of compatible shapes
template <typename T>
static mlir::LogicalResult verifyCompatibleOperandsAndResults(T* op, bool checkResult = true)
{
    mlir::Type t0 = op->getOperation()->getOperand(0)->getType();
    mlir::NGTensorType opType0 = t0.cast<NGTensorType>();

    Operation* opr = op->getOperation();
    auto i = 0;
    for (auto operand : opr->getOperands())
    {
        if (i == 0)
            continue;
        mlir::Type t = operand->getType();
        mlir::NGTensorType opType = t.cast<NGTensorType>();
        if (!opType.isCompatible(opType0))
            return op->emitOpError("Incompatible operand shape");
        i++;
    }

    if (checkResult)
    {
        for (auto result : opr->getResults())
        {
            mlir::Type t = result->getType();
            mlir::NGTensorType resType = t.cast<NGTensorType>();
            if (!resType.isCompatible(opType0))
                return op->emitOpError("Incompatible operand shape");
        }
    }
    return mlir::success();
}

template <typename T>
static mlir::LogicalResult verifyUnaryArithOp(T* op)
{
    return verifyCompatibleOperandsAndResults(op);
}

template <typename T>
static mlir::LogicalResult verifyBinaryArithOp(T* op)
{
    return verifyCompatibleOperandsAndResults(op);
}

template <typename T>
static mlir::LogicalResult verifyAxisReductionOp(T* op)
{
    return mlir::failure();
}

template <typename T>
static mlir::LogicalResult verifyLogicalReductionOp(T* op)
{
    // TODO: verifyAxisReductionOp(op) + input and return element type.
    return mlir::failure();
}

template <typename T>
static mlir::LogicalResult verifyIndexReductionOp(T* op)
{
    // TODO: verifyAxisReductionOp(op) + return element type + single axis.
    return mlir::success();
}

template <typename T>
static mlir::LogicalResult verifyOp(T* op)
{
    return op->emitOpError("Unsupported verifier for this operation");
}

template <>
mlir::LogicalResult verifyOp(NGDotOp* op)
{
    // TODO(dcab): Improve verification: proper shapes, etc.
    return mlir::success();
}

template <>
mlir::LogicalResult verifyOp(NGConcatOp* op)
{
    // TODO(amprocte): Improve verification: proper shapes, etc.
    return mlir::success();
}

template <>
mlir::LogicalResult verifyOp(NGSelectOp* op)
{
    mlir::Type t0 = op->getOperation()->getOperand(0)->getType();
    mlir::Type t1 = op->getOperation()->getOperand(1)->getType();
    mlir::Type t2 = op->getOperation()->getOperand(2)->getType();
    mlir::Type r0 = op->getOperation()->getResult(0)->getType();

    NGTensorType opType0 = t0.cast<NGTensorType>();
    NGTensorType opType1 = t1.cast<NGTensorType>();
    NGTensorType opType2 = t2.cast<NGTensorType>();
    NGTensorType resType = r0.cast<NGTensorType>();

    // arg1 arg2 of same shape and elt type
    if (!opType1.isCompatible(opType2))
        return op->emitOpError("Incompatible operand shapes or types for select op");
    // arg0 of same shape and elt type is bool
    if (!opType0.isCompatibleShape(opType1) || !opType0.getElementType().isa<NGBoolType>())
        return op->emitOpError("Incompatible shape for arg0 of select op");
    // result is of same shape and elt type as arg1/2
    if (!resType.isCompatible(opType1))
        return op->emitOpError("Incompatible result shape or type for select op");

    return mlir::success();
}

template <typename T>
static mlir::LogicalResult verifyCmpOp(T* op)
{
    mlir::LogicalResult result = verifyCompatibleOperandsAndResults(op, false /*checkResult*/);
    if (failed(result))
    {
        return result;
    }

    mlir::Type t0 = op->getOperation()->getOperand(0)->getType();
    mlir::NGTensorType opType0 = t0.cast<NGTensorType>();

    mlir::Type r0 = op->getOperation()->getResult(0)->getType();
    NGTensorType resType = r0.cast<NGTensorType>();

    // result of same shape as input and has bool type
    if (!resType.isCompatibleShape(opType0) || !resType.getElementType().isa<NGBoolType>())
        return op->emitOpError("Incompatible result shape or type for comparison op");

    return mlir::success();
}

template <>
mlir::LogicalResult verifyOp(NGGatherOp* op)
{
    Type ty = op->params()->getType();
    NGTensorType inputType = ty.cast<NGTensorType>();

    ty = op->indices()->getType();
    NGTensorType indicesType = ty.cast<NGTensorType>();

    // ensure axis < params rank
    if (op->axis().getSExtValue() >= inputType.getRank())
        return op->emitOpError("Gather axis is larger than input rank");

    ty = indicesType.getElementType();

    // ensure indices are I32 or I64
    if (!ty.isa<NGIntegerType>())
        return op->emitOpError("Indices tensor is not of Integer type");

    NGIntegerType indicesEltType = ty.cast<NGIntegerType>();
    if (!indicesEltType.isInt32() && !indicesEltType.isInt64())
        return op->emitOpError("Indices tensor is not of I32 or I64 type");

    mlir::Type r0 = op->res()->getType();
    NGTensorType resType = r0.cast<NGTensorType>();

    // ensure result is compatible with input
    if (!resType.getRank() == inputType.getRank() + indicesType.getRank() - 1)
        return op->emitOpError("Incompatible result shape and/or type");

    return mlir::success();
}

namespace mlir
{
#define GET_OP_CLASSES
#include "ops.cpp.inc"
}
