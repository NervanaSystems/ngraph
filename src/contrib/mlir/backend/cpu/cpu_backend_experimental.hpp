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
//
// This file contains code that is temporarily needed to overcome existing limitations in MLIR.
//
// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#pragma once

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>

namespace mlir
{
    class MemRefDescriptor;
}

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            /// Custom Std-to-LLVM type converter that overrides `convertType` and
            /// `convertFunctionSignature` taking into account that MemRef type will be lowered to a
            /// plain pointer. It falls back to the standar LLVMTypeConverter for the remaining
            /// types.
            class CustomLLVMTypeConverter : public mlir::LLVMTypeConverter
            {
            public:
                using LLVMTypeConverter::LLVMTypeConverter;

                /// Converts MemRef type to a plain LLVM pointer to element type. Falls back to
                /// default LLVMTypeConverter for other types.
                mlir::Type convertType(mlir::Type type) override;

                /// Converts function signature following LLVMTypeConverter approach but lowering
                /// MemRef arguments to plain LLVM pointers to element type.
                mlir::LLVM::LLVMType convertFunctionSignature(
                    mlir::FunctionType type,
                    bool isVariadic,
                    mlir::LLVMTypeConverter::SignatureConversion& result) override;

                /// Create a DefaultMemRefDescriptor object for 'value'.
                std::unique_ptr<mlir::MemRefDescriptor>
                    createMemRefDescriptor(mlir::Value* value) override;

                /// Builds IR creating a nullptr value of the descriptor type.
                std::unique_ptr<mlir::MemRefDescriptor>
                    buildMemRefDescriptor(mlir::OpBuilder& builder,
                                          mlir::Location loc,
                                          mlir::Type descriptorType) override;
                /// Builds IR creating a MemRef descriptor that represents `type` and populates it
                /// with static shape and stride information extracted from the type.
                std::unique_ptr<mlir::MemRefDescriptor>
                    buildStaticMemRefDescriptor(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::MemRefType type,
                                                mlir::Value* memory) override;

            private:
                mlir::Type convertMemRefType(mlir::MemRefType type);
            };

            /// Populates 'patterns' with default LLVM conversion patterns using
            /// CustomLLVMTypeConverter and a custom conversion pattern for FuncOp which takes into
            /// account MemRef custom lowering to plain LLVM pointer.
            void
                customPopulateStdToLLVMConversionPatterns(mlir::LLVMTypeConverter& converter,
                                                          mlir::OwningRewritePatternList& patterns);
        } // namespace ngmlir
    }     // namespace runtime
} // namespace ngraph
