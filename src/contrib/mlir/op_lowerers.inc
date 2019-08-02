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

// Add new dialect ops with lowering support to this file
#ifndef MLIR_OP
#define MLIR_OP
#endif

#ifndef MLIR_LAST_OP
#define MLIR_LAST_OP(OP) MLIR_OP(OP)
#endif

MLIR_OP(NGAddOp)
MLIR_OP(NGArgMaxRedOp)
MLIR_OP(NGArgMinRedOp)
MLIR_OP(NGConcatOp)
MLIR_OP(NGDivOp)
MLIR_OP(NGDotOp)
MLIR_OP(NGGatherOp)
MLIR_OP(NGGreaterOp)
MLIR_OP(NGLessOp)
MLIR_OP(NGMulOp)
MLIR_OP(NGMaxOp)
MLIR_OP(NGMinOp)
MLIR_OP(NGReluOp)
MLIR_OP(NGSubOp)
MLIR_LAST_OP(NGReturnOp)

#undef MLIR_OP
#undef MLIR_LAST_OP
