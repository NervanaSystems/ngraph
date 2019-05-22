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
        namespace ngmlir
        {
            // TODO:
            // - Move verifiers and other OP helpers (e.g. getSomeAttribute()) to separate files
            //
            // - Op helpers: Since it is not possible to add arbitrary code (and would complicate the .td file)
            // to Ops classes, we will add helper classes with static methods for each Op that needs it

            // Additional verification methods
            // Tensor type checks are already verified by the caller of these methods
            template <typename T>
            static mlir::LogicalResult verifyUnaryArithOp(T* op)
            {
                // TODO: Check matching element types
                return mlir::success();
            }

            // Additional verification methods
            // Tensor type checks are already verified by the caller of these methods
            template <typename T>
            static mlir::LogicalResult verifyBinaryArithOp(T* op)
            {
                // TODO: Check matching element types
                return mlir::success();
            }

            template <typename T>
            static mlir::LogicalResult verifyOp(T* op)
            {
                return op->emitOpError("Unsupported verifier for this operation");
            }

            // Per op specializations
            template <>
            mlir::LogicalResult verifyOp<NGMatMulBiasOp>(NGMatMulBiasOp* op)
            {
                // Verify that we have 2 operands
                // Bias operand must be null for now (not implemented)
                if (op->getNumOperands() != 2)
                {
                    std::stringstream ss;
                    ss << "Unexpected MatmulBiasOp with " << op->getNumOperands()
                       << " operands. 3 operands expected";
                    return op->emitOpError(ss.str());
                }

                // Verify that operand types are supported.
                auto op0_tensor_ty = op->getOperand(0)->getType().cast<NGTensorType>();
                auto op1_tensor_ty = op->getOperand(1)->getType().cast<NGTensorType>();

                // Verify that operand shapes are supported.
                if (op0_tensor_ty.getRank() != 2 || op1_tensor_ty.getRank() != 2)
                {
                    return op->emitOpError(
                        "Unsupported number of dimensions. Only 2D tensors are supported in "
                        "MatmulBiasOp");
                }

                // TODO(dcab): Improve verification: matching types, proper shapes, etc.

                return mlir::success();
            }
        }
    }

    using namespace mlir;
    namespace runtime
    {
        namespace ngmlir
        {
#define GET_OP_CLASSES
#include "ops.cpp.inc"
        }
    }
}
