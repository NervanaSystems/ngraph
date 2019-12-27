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
// This is the nGraph Dialect operation definition file.
//
//===----------------------------------------------------------------------===//

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

include "mlir/IR/OpBase.td"
include "core/ngraph_dialect/ops.td"

// nGraph Dialect pattern match definitions for fused Op's
//
// This files declares nGraph fused operations that table-gen uses to create C++ code
// For more information about tablegen. See https://llvm.org/docs/TableGen/index.html
//
// The output files are fused_ops.h.inc and fused_ops.cpp.inc and are generated at build time
// The file declares base classes to ease opcode definitions and hoist common parts out.
// Each class fixes a set of attributes. For example:
//
// Each def will corresponding to a C++ class

// class for defining the pattern to fuse MatMul + Bias

def MatMulBiasPattern : Pat<(NGAddOp ( NGDotOp $input1, $input2), $add_input), (NGMatMulBiasOp $input1, $input2, $add_input)>;
