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

#include "dialect.hpp"
#include "ngraph/check.hpp"
#include "ops.hpp"
#include "type.hpp"

using namespace mlir;

NGDialect::NGDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect("ng", ctx)
{
    addTypes<NGTensorType>();
    addTypes<NGIntegerType>();
    addTypes<NGBoolType>();

    addOperations<
#define GET_OP_LIST
#include "ops.cpp.inc"
        >();
}

void NGDialect::printType(mlir::Type type, raw_ostream& os) const
{
    switch (type.getKind())
    {
    case NG_TENSOR_TYPE_ID:
    {
        os << "tensor<";
        auto tensor_ty = type.cast<NGTensorType>();
        for (auto dim : tensor_ty.getShape())
        {
            os << dim << 'x';
        }
        os << tensor_ty.getElementType() << '>';
        return;
    }
    case NG_I8_TYPE_ID:
    case NG_I16_TYPE_ID:
    case NG_I32_TYPE_ID:
    case NG_I64_TYPE_ID:
    case NG_U8_TYPE_ID:
    case NG_U16_TYPE_ID:
    case NG_U32_TYPE_ID:
    case NG_U64_TYPE_ID:
    {
        auto int_ty = type.cast<NGIntegerType>();
        os << "i" << int_ty.getWidth();
        return;
    }
    case NG_BOOL_TYPE_ID:
    {
        os << "bool";
        return;
    }
    default: { NGRAPH_CHECK(false, "Incorrect type to print?");
    }
    }
}
