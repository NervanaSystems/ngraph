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

#include "type.hpp"

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "ngraph/assertion.hpp"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using namespace mlir;

unsigned NGIntegerType::getWidth() const
{
    switch (getKind())
    {
    case NG_I8_TYPE_ID:
    case NG_U8_TYPE_ID: return 8;
    case NG_I16_TYPE_ID:
    case NG_U16_TYPE_ID: return 16;
    case NG_I32_TYPE_ID:
    case NG_U32_TYPE_ID: return 32;
    case NG_I64_TYPE_ID:
    case NG_U64_TYPE_ID: return 64;
    default: NGRAPH_CHECK(false, "Invalid type ID");
    }
    return 0;
}

bool NGIntegerType::isSigned() const
{
    switch (getKind())
    {
    case NG_I8_TYPE_ID:
    case NG_I16_TYPE_ID:
    case NG_I32_TYPE_ID:
    case NG_I64_TYPE_ID: return true;
    case NG_U8_TYPE_ID:
    case NG_U16_TYPE_ID:
    case NG_U32_TYPE_ID:
    case NG_U64_TYPE_ID: return false;
    default: NGRAPH_CHECK(false, "Invalid type ID");
    }
    return false;
}

/// Creates TensorType objects. They all point to the same storage if
/// element type and shape are the same.
NGTensorType NGTensorType::get(MLIRContext* context, EltType eltType, Shape shape)
{
    return Base::get(context, NGTypeKind::NG_TENSOR_TYPE_ID, eltType, shape);
}

bool NGTensorType::isCompatible(NGTensorType& other) const
{
    // Exact same tensor
    if (this == &other)
        return true;
    // different tensors, check if of same element type and compatible shapes
    if (getElementType() != other.getElementType())
        return false;
    // TODO: Handle dynamic ranks
    // MLIR MemRefType doesn't seem to support it at the moment.
    return isCompatibleShape(other);
}

bool NGTensorType::isCompatibleShape(NGTensorType& other) const
{
    auto shape = getShape();
    auto otherShape = other.getShape();

    if (shape.size() != otherShape.size())
        return false;

    for (auto i = 0; i < shape.size(); i++)
    {
        NGRAPH_CHECK(shape[i] >= -1, "Invalid tensor shape", shape[i]);
        NGRAPH_CHECK(otherShape[i] >= -1, "Invalid tensor shape", otherShape[i]);

        if (shape[i] == -1 || otherShape[i] == -1 || shape[i] == otherShape[i])
            continue;
        return false;
    }
    return true;
}
