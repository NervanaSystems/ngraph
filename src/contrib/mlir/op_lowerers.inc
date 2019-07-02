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


// Add new dialect ops lowerers to this file

#define DECL_OP_CONV(OP)                                                                           \
    class OP##Conversion : public NGraphOpLowering                                                 \
    {                                                                                              \
    public:                                                                                        \
        explicit OP##Conversion(mlir::MLIRContext* context, DialectLoweringPass& pass)             \
            : NGraphOpLowering(mlir::OP::getOperationName(), context, pass)                        \
        {                                                                                          \
        }                                                                                          \
                                                                                                   \
        PatternMatchResult matchAndRewrite(Operation* op,                                          \
                                           ArrayRef<Value*> operands,                              \
                                           PatternRewriter& rewriter) const override;              \
    };

DECL_OP_CONV(NGAddOp)
DECL_OP_CONV(NGDotOp)
DECL_OP_CONV(NGReturnOp)

#undef DECL_OP_CONV
