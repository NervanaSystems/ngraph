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
        {
            continue;
        }
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
    if (resType.getRank() != inputType.getRank() + indicesType.getRank() - 1)
        return op->emitOpError("Incompatible result shape and/or type");

    return mlir::success();
}

template <>
mlir::LogicalResult verifyOp(NGConvolutionOp* op)
{
    Type ty = op->images()->getType();
    NGTensorType imagesType = ty.cast<NGTensorType>();
    Type imagesEt = imagesType.getElementType();
    Shape imagesShape = imagesType.getShape();

    ty = op->filters()->getType();
    NGTensorType filtersType = ty.cast<NGTensorType>();
    Type filtersEt = filtersType.getElementType();
    Shape filtersShape = filtersType.getShape();

    ty = op->res()->getType();
    NGTensorType resultType = ty.cast<NGTensorType>();
    Type resultEt = resultType.getElementType();
    Shape resultShape = resultType.getShape();

    ArrayAttr strides = op->strides();
    ArrayAttr padBelow = op->padBelow();
    ArrayAttr padAbove = op->padAbove();

    unsigned imagesRank = imagesShape.size();
    unsigned filtersRank = filtersShape.size();
    unsigned resultRank = resultShape.size();
    unsigned imageSpatialRank = imagesRank - 2;
    unsigned filtersSpatialRank = filtersRank - 2;
    unsigned stridesRank = strides.size();
    unsigned padBelowRank = padBelow.size();
    unsigned padAboveRank = padAbove.size();

    SmallVector<int64_t, 4> stridesVal, padAboveVal, padBelowVal;
    // Identical filters and image element types
    if (filtersEt != imagesEt)
    {
        return op->emitOpError("Incompatible image and filters types");
    }

    // Verify image shape
    if (imagesRank < 3)
    {
        return op->emitOpError("Image shape of rank below 3");
    }

    // Verify strides and pads shapes
    if (imageSpatialRank != stridesRank || imageSpatialRank != padBelowRank ||
        imageSpatialRank != padAboveRank)
    {
        return op->emitOpError("Image spatial rank mismatches strides and/or padding ranks");
    }

    if (imageSpatialRank != filtersSpatialRank)
    {
        return op->emitOpError("Image and filters spatial ranks mismatch");
    }

    // Batch size is non-zero, and identical non-zero channel depth
    if (imagesShape[0] <= 0 || filtersShape[0] <= 0 || imagesShape[1] != filtersShape[1] ||
        imagesShape[1] <= 0)
    {
        return op->emitOpError("Image and filters have invalid shapes");
    }

    for (auto attrs : llvm::zip(strides, padBelow, padAbove))
    {
        auto s = std::get<0>(attrs).cast<IntegerAttr>().getInt();
        auto pb = std::get<1>(attrs).cast<IntegerAttr>().getInt();
        auto pa = std::get<2>(attrs).cast<IntegerAttr>().getInt();
        if (s <= 0)
        {
            return op->emitOpError("Window stride must be non-negative");
        }
        if (pb < 0 || pa < 0)
        {
            return op->emitOpError("Paddings must be positive");
        }
        stridesVal.push_back(s);
        padBelowVal.push_back(pb);
        padAboveVal.push_back(pa);
    }

    // Check output shape
    if (resultRank != imagesRank || resultShape[0] != imagesShape[0] ||
        resultShape[1] != filtersShape[0])
    {
        return op->emitOpError("Invalid result shape");
    }
    for (unsigned i = 0; i < resultRank - 2; i++)
    {
        unsigned resDim = llvm::divideCeil(padBelowVal[i] + padAboveVal[i] + imagesShape[2 + i] -
                                               filtersShape[2 + i] + 1,
                                           stridesVal[i]);
        if (resultShape[2 + i] != resDim)
        {
            return op->emitOpError("Invalid result spatial shape");
        }
    }
    return mlir::success();
}

static std::string getBufferIdAttrName()
{
    return "ng.buffer_id";
}

void setBufferId(mlir::Operation* op, mlir::IntegerAttr attr)
{
    op->setAttr(getBufferIdAttrName(), attr);
}

mlir::IntegerAttr setBufferId(mlir::Operation* op, unsigned val)
{
    auto attr = mlir::IntegerAttr::get(IntegerType::get(32, op->getContext()), val);
    setBufferId(op, attr);
    return attr;
}

mlir::IntegerAttr getBufferId(mlir::Operation* op)
{
    return op->getAttrOfType<mlir::IntegerAttr>(getBufferIdAttrName());
}

namespace mlir
{
#define GET_OP_CLASSES
#include "ops.cpp.inc"
}
