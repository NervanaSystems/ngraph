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

// NOTE: This file follows nGraph format style and MLIR naming convention since
// it does
// not expose public API to the rest of nGraph codebase and heavily depends on
// MLIR API.

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "ngraph/check.hpp"
namespace mlir
{
    using llvm::raw_ostream;

    // reuse std float types as-is
    using NGFloatType = mlir::FloatType;
    using NGIntegerType = mlir::IntegerType;

    /// Boolean Type.
    class NGBoolType : public mlir::Type::TypeBase<NGBoolType, mlir::Type, mlir::TypeStorage>
    {
    public:
        using Base::Base;

        static NGBoolType get(mlir::MLIRContext* ctx) { return get(ctx); }
        size_t getWidth() { return 8; }
    };

    // Note that dialect types don't add new data members, so always possible
    // to use NG or std types here
    using EltType = mlir::Type;
    // TODO: Can we use ngraph::shape here (given the hashing requirements)
    using Shape = llvm::ArrayRef<int64_t>;

    /// Tensor Type storage. There is a unique instance per type attributes.
    /// Tensor Type is combination of the element type and shape. Each different
    /// shape is a unique type.
    struct NGTensorTypeStorage : public mlir::TypeStorage
    {
        // Tensor key is its type and shape.
        // This is called when the user requests a specific tensor type
        using KeyTy = std::tuple<EltType, Shape>;

        static unsigned hashKey(const KeyTy& key)
        {
            return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
        }

        bool operator==(const KeyTy& key) const
        {
            return key == KeyTy(getElementType(), getShape());
        }

        static NGTensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                              const KeyTy& key)
        {
            // Deep copy the type shape over to MLIR context
            EltType eltType = std::get<0>(key);
            Shape shape = allocator.copyInto(std::get<1>(key));
            auto* storage = allocator.allocate<NGTensorTypeStorage>();
            return new (storage) NGTensorTypeStorage(eltType, shape);
        }

        Shape getShape() const { return shape; }
        int64_t getRank() const { return shape.size(); }
        EltType getElementType() const { return eltType; }

    private:
        NGTensorTypeStorage(EltType eltType, Shape shape)
            : eltType(eltType)
            , shape(shape)
        {
        }

    private:
        EltType eltType;
        Shape shape;
    };

    /// NGraph Tensor Type
    class NGTensorType : public mlir::Type::TypeBase<NGTensorType, mlir::Type, NGTensorTypeStorage>
    {
    public:
        using Base::Base;
        EltType getElementType() const { return getImpl()->getElementType(); }
        Shape getShape() const { return getImpl()->getShape(); }
        /// Tensor Rank. Static shape only for now
        int getRank() { return getShape().size(); }
        /// Computes tensor size in bytes
        size_t getSizeInBytes()
        {
            return getNumElements() * llvm::divideCeil(getElementBitWidth(), 8);
        }
        size_t getElementBitWidth()
        {
            Type type = getElementType();
            if (NGIntegerType intType = type.dyn_cast<NGIntegerType>())
                return intType.getWidth();
            if (NGFloatType floatType = type.dyn_cast<NGFloatType>())
                return floatType.getIntOrFloatBitWidth();
            if (NGBoolType boolType = type.dyn_cast<NGBoolType>())
                return boolType.getWidth();
            NGRAPH_CHECK(false, "Unknown type");
            return -1;
        }
        /// Get number of elements
        size_t getNumElements()
        {
            size_t s = 1;
            auto shape = getShape();
            for (auto i = 0; i < getRank(); i++)
            {
                // no dynamic dims
                if (shape[i] == -1)
                    return -1;
                s *= shape[i];
            }
            return s;
        }
        /// Checks if two tensors are compatible. Compatible means:
        /// Exactly same element types
        /// Compatible shapes: see isCompatibleShape.
        bool isCompatible(NGTensorType& other) const;

        /// Check if Shapes are of same rank and  matching dimensions unless one of
        /// them is dynamic.
        bool isCompatibleShape(NGTensorType& other) const;

        /// create a unique tensor type based on element type and shape.
        static NGTensorType get(mlir::MLIRContext* context, EltType eltType, Shape shape);
    };
}
