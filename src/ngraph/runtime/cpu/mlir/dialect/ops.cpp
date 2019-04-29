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

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <typename T>
            static mlir::LogicalResult verifyBinOperands(T* op)
            {
                if (!op->getOperand(0)->getType().template isa<NGTensorType>())
                {
                    std::string msg;
                    raw_string_ostream os(msg);
                    os << "expects a Tensor type for LHS, got " << op->getOperand(0)->getType();
                    return op->emitOpError(os.str());
                }
                if (!op->getOperand(1)->getType().template isa<NGTensorType>())
                {
                    std::string msg;
                    raw_string_ostream os(msg);
                    os << "expects a Tensor type for RHS, got " << op->getOperand(0)->getType();
                    return op->emitOpError(os.str());
                }
                return mlir::success();
            }

            template <typename T>
            static mlir::LogicalResult verifySingleOperand(T* op)
            {
                if (!op->getOperand()->getType().template isa<NGTensorType>())
                {
                    std::string msg;
                    raw_string_ostream os(msg);
                    os << "expects a Tensor Type for its argument, got "
                       << op->getOperand()->getType();
                    return op->emitOpError(os.str());
                }
                return mlir::success();
            }
        }
    }

    void runtime::cpu::NG_FakeOutput::build(mlir::Builder* builder,
                                            mlir::OperationState* state,
                                            mlir::Type resultType)
    {
        state->types.push_back(std::move(resultType));
    }

    mlir::LogicalResult runtime::cpu::NG_FakeOutput::verify()
    {
        // TODO: Verify returned tensor types must match function return type.
        return mlir::success();
    }

    void runtime::cpu::NG_AddOp::build(mlir::Builder* builder,
                                       mlir::OperationState* state,
                                       mlir::Value* lhs,
                                       mlir::Value* rhs)
    {
        state->types.push_back(lhs->getType());
        state->operands.push_back(lhs);
        state->operands.push_back(rhs);
    }

    mlir::LogicalResult runtime::cpu::NG_AddOp::verify()
    {
        // TODO: verify matching elt types
        verifyBinOperands(this);
        return mlir::success();
    }

    void runtime::cpu::NG_ReturnOp::build(mlir::Builder* builder,
                                          mlir::OperationState* state,
                                          std::vector<mlir::Value*> value_list)
    {
        for (auto value : value_list)
        {
            if (value)
                state->operands.push_back(value);
        }
    }

    mlir::LogicalResult runtime::cpu::NG_ReturnOp::verify()
    {
        // TODO: Verify returned tensor types must match function return type.
        return mlir::success();
    }
}
