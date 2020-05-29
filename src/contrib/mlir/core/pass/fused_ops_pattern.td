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
//===----------------------------------------------------------------------===//
//
// nGraph Dialect pattern match definitions for fused Op's using DRR
//
// This files declares nGraph fused operations that table-gen uses to create 
// C++ code. For more information about tablegen, 
// See https://llvm.org/docs/TableGen/index.html
//
// The output file fused_ops_pattern.h.inc is generated at build time
// Each def will corresponding to a C++ class

// NOTE: This file follows nGraph format style and MLIR naming convention since 
// it does not expose public API to the rest of nGraph codebase and heavily 
// depends on MLIR API.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "core/ngraph_dialect/ops.td"

// Native code call to c++ helper funnction to create SGEMM Op
def createSgemmOp :  NativeCodeCall<"createSgemmOp($_builder, $0.getDefiningOp(), $1, $2, $3)">;

// class for defining the pattern to fuse MatMul + Bias
def MatMulBiasPattern : Pat<(NGAddOp:$old_op ( NGDotOp $input1, $input2), $add_input), 
			    (createSgemmOp $old_op, $input1, $input2, $add_input)>;
