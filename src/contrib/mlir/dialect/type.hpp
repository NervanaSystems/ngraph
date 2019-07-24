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

    enum NGTypeKind
    {
        // The enum starts at the range reserved for this dialect.
        // These values are pre-defined in MLIR lib and not configurable from here.
        NG_TYPE = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
        // Element types that are added by the dialect.
        // Other types are just re-use of std dialect types.
        NG_FIRST_INT_TYPE_ID,
        NG_I8_TYPE_ID = NG_FIRST_INT_TYPE_ID,
        NG_I16_TYPE_ID,
        NG_I32_TYPE_ID,
        NG_I64_TYPE_ID,
        NG_U8_TYPE_ID,
        NG_U16_TYPE_ID,
        NG_U32_TYPE_ID,
        NG_U64_TYPE_ID,
        NG_LAST_INT_TYPE_ID = NG_U64_TYPE_ID,
        NG_BOOL_TYPE_ID,
        // Tensor type
        NG_TENSOR_TYPE_ID
    };

    // reuse std float types as-is
    using NGFloatType = mlir::FloatType;

    /// Integer type. It represents an integer of width 8,16,32,64. Signed or not.
    class NGIntegerType : public mlir::Type::TypeBase<NGIntegerType, mlir::Type>
    {
    public:
        using Base::Base;

        static NGIntegerType get(NGTypeKind kind, mlir::MLIRContext* context)
        {
            NGRAPH_CHECK(kindof(kind), "Not an integer kind.");
            return Base::get(context, kind);
        }
        /// Create signed Int8
        static NGIntegerType getInt8(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_I8_TYPE_ID, ctx);
        }
        /// Create signed Int16
        static NGIntegerType getInt16(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_I16_TYPE_ID, ctx);
        }
        /// Create signed Int32
        static NGIntegerType getInt32(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_I32_TYPE_ID, ctx);
        }
        /// Create signed Int64
        static NGIntegerType getInt64(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_I64_TYPE_ID, ctx);
        }
        /// Create unsigned Int8
        static NGIntegerType getUInt8(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_U8_TYPE_ID, ctx);
        }
        /// Create unsigned Int16
        static NGIntegerType getUInt16(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_U16_TYPE_ID, ctx);
        }
        /// Create unsigned Int32
        static NGIntegerType getUInt32(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_U32_TYPE_ID, ctx);
        }
        /// Create unsigned Int64
        static NGIntegerType getUInt64(mlir::MLIRContext* ctx)
        {
            return get(NGTypeKind::NG_U64_TYPE_ID, ctx);
        }

        /// RTTI support. So we can do obj->isa<NGIntegerType>()
        static bool kindof(unsigned kind)
        {
            return kind >= NGTypeKind::NG_FIRST_INT_TYPE_ID &&
                   kind <= NGTypeKind::NG_LAST_INT_TYPE_ID;
        }

        /// Return the bitwidth of this integer type.
        unsigned getWidth() const;

        /// Check if signed type
        bool isSigned() const;

        /// Check if Int8
        bool isInt8() const { return getKind() == NG_I8_TYPE_ID; }
        /// Check if UInt8
        bool isUInt8() const { return getKind() == NG_U8_TYPE_ID; }
        /// Check if Int16
        bool isInt16() const { return getKind() == NG_I16_TYPE_ID; }
        /// Check if UInt16
        bool isUInt16() const { return getKind() == NG_U16_TYPE_ID; }
        /// Check if Int32
        bool isInt32() const { return getKind() == NG_I32_TYPE_ID; }
        /// Check if UInt32
        bool isUInt32() const { return getKind() == NG_U32_TYPE_ID; }
        /// Check if Int64
        bool isInt64() const { return getKind() == NG_I64_TYPE_ID; }
        /// Check if UInt64
        bool isUInt64() const { return getKind() == NG_U64_TYPE_ID; }
        // Delete convenience methods inherited from MLIR Type class.
        // This would avoid confusion if we do something like this and get false.
        //
        //      if (type->cast<NGIntegerType>()->isInteger(32)) {}
        //
        // Those helpers use type id, and since we have our own Integer type id, they
        // don't apply.
        bool isInteger(unsigned width) const = delete;
        unsigned getIntOrFloatBitWidth() const = delete;
        bool isIntOrIndex() const = delete;
        bool isIntOrIndexOrFloat() const = delete;
        bool isIntOrFloat() const = delete;
    };

    /// Boolean Type.
    class NGBoolType : public mlir::Type::TypeBase<NGBoolType, mlir::Type>
    {
    public:
        using Base::Base;
        static NGBoolType get(NGTypeKind kind, mlir::MLIRContext* context)
        {
            NGRAPH_CHECK(kindof(kind), "Not a bool type.");
            return Base::get(context, kind);
        }

        static bool kindof(unsigned kind) { return kind == NGTypeKind::NG_BOOL_TYPE_ID; }
        static NGBoolType get(mlir::MLIRContext* ctx) { return get(NG_BOOL_TYPE_ID, ctx); }
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

        Shape getShape() const { return m_shape; }
        int64_t getRank() const { return m_shape.size(); }
        EltType getElementType() const { return m_eltType; }
    private:
        NGTensorTypeStorage(EltType eltType, Shape shape)
            : m_eltType(eltType)
            , m_shape(shape)
        {
        }

    private:
        EltType m_eltType;
        Shape m_shape;
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

        /// Check if Shapes are of same rank and  matching dimensions unless one of them is dynamic.
        bool isCompatibleShape(NGTensorType& other) const;

        /// create a unique tensor type based on element type and shape.
        static NGTensorType get(mlir::MLIRContext* context, EltType eltType, Shape shape);

        /// for llvm RTTI
        static bool kindof(unsigned kind) { return kind == NGTypeKind::NG_TENSOR_TYPE_ID; }
    };
}
