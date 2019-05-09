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

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            using llvm::raw_ostream;

            enum NGTypeKind
            {
                // The enum starts at the range reserved for this dialect.
                // These values are pre-defined in MLIR lib and not configurable from here.
                NG_TYPE = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
                TENSOR_TYPE_ID
            };

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

            class NGTensorType
                : public mlir::Type::TypeBase<NGTensorType, mlir::Type, NGTensorTypeStorage>
            {
            public:
                using Base::Base;
                EltType getElementType() const { return getImpl()->getElementType(); }
                Shape getShape() const { return getImpl()->getShape(); }
                int getRank() { return getShape().size(); }
                size_t getSizeInBytes()
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
                    // Multiply times element size
                    return s * llvm::divideCeil(getElementType().getIntOrFloatBitWidth(), 8);
                }
                /// convert to memref native MLIR type. Used for lowering.
                mlir::MemRefType toMemref();
                /// create a unique tensor type based on element type and shape.
                static NGTensorType get(mlir::MLIRContext* context, EltType eltType, Shape shape);
                /// for llvm RTTI
                static bool kindof(unsigned kind) { return kind == NGTypeKind::TENSOR_TYPE_ID; }
            };
        }
    }
}
