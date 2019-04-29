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

#include <cstdarg>
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            // Fake instructions
            class NG_FakeOutput : public mlir::Op<NG_FakeOutput,
                                                  mlir::OpTrait::NOperands<0>::Impl,
                                                  mlir::OpTrait::OneResult,
                                                  mlir::OpTrait::HasNoSideEffect>
            {
            public:
                static llvm::StringRef getOperationName() { return "ng.fake.output"; }
                mlir::LogicalResult verify();
                static void
                    build(mlir::Builder* builder, mlir::OperationState* state, mlir::Type type);
                /// Inherit constructor.
                using Op::Op;
            };

            // Binary instructions
            class NG_AddOp : public mlir::Op<NG_AddOp,
                                             mlir::OpTrait::NOperands<2>::Impl,
                                             mlir::OpTrait::OneResult,
                                             mlir::OpTrait::HasNoSideEffect>
            {
            public:
                static llvm::StringRef getOperationName() { return "ng.add"; }
                /// custom verification
                mlir::LogicalResult verify();
                static void build(mlir::Builder* builder,
                                  mlir::OperationState* state,
                                  mlir::Value* lhs,
                                  mlir::Value* rhs);

                /// Convenience accessor for LHS of the expression.
                mlir::Value* getLHS() { return getOperand(0); }
                /// Convenience accessor for RHS of the expression.
                mlir::Value* getRHS() { return getOperand(1); }
                /// Inherit constructor.
                using Op::Op;
            };

            /// Return operations terminate blocks (and functions as well). They take a
            /// single argument and the type must match the function return type.
            class NG_ReturnOp : public mlir::Op<NG_ReturnOp,
                                                mlir::OpTrait::VariadicOperands,
                                                mlir::OpTrait::ZeroResult,
                                                mlir::OpTrait::IsTerminator>
            {
            public:
                static llvm::StringRef getOperationName() { return "ng.return"; }
                /// Operations can add custom verification beyond the traits they define.
                mlir::LogicalResult verify();

                /// Interface to mlir::Builder::create<PrintOp>(...)
                /// This method populate the `state` that MLIR use to create operations.
                /// The `toy.return` operation accepts an optional single array as an argument
                /// and does not have any returned value.
                static void build(mlir::Builder* builder,
                                  mlir::OperationState* state,
                                  std::vector<mlir::Value*> value_list);

                /// Return true if there is a returned value.
                bool hasOperand() { return 0 != getNumOperands(); }
                /// Helper to return the optional operand. Caller must check if the operand
                /// is present before calling this.
                mlir::Value* getOperand() { return getOperation()->getOperand(0); }
                mlir::Value* getOperand(unsigned i) { return getOperation()->getOperand(i); }
                /// Inherit constructor.
                using Op::Op;
            };
        }
    }
}