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
include "core/ngraph_dialect/ops_interfaces.td"
// nGraph Dialect operations definitions
//
// This files declares nGraph operations that table-gen uses to create C++ code
// For more information about tablegen. See https://llvm.org/docs/TableGen/index.html
//
// The output files are ops.h.inc and ops.cpp.inc and are generated at build time
// The file declares base classes to ease opcode definitions and hoist common parts out.
// Each class fixes a set of attributes. For example:
// class NG_Unary_Arith_Op defines a base class for all unary arithmetic ops without side-effects
//
// An opcode is a record definition of the form
//      def AbsOp      : NG_Unary_Arith_Op<"abs">;
//
// Each def will corresponding to a C++ class

def NG_Dialect : Dialect {
  let name = "ng";
  // TODO: Have the dialect under its own mlir::ngraph namespace
  // At mlir top-level for now
  let cppNamespace = "";
}

// nGraph Types
// This defines records equivalent to nGraph types. It doesn't generate code.
// This is used as a type in the DAG input/outputs.
// Constraints (CPred) are used to type-check args/results of that type during op verification
def NG_TensorType : Type<CPred<"$_self.isa<mlir::NGTensorType>()">,
                     "nGraph Tensor Type">;

// A generic un-typed MemRef. Used for Fake instructions inserted during dialect lowering
def NG_MemRefType : Type<IsMemRefTypePred, "MemRef Type">;

// nGraph operation base class.
// Prepends "ng." to operation name
class NG_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<NG_Dialect, mnemonic, traits> {}

// Operations producing single result.
// Will set OneResult trait based on Results out dag.
class NG_OneResult_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs NG_TensorType:$res)> {}

// Operations producing no results
class NG_ZeroResult_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs)> {}

// Base class for arithmetic unary operations without side effects.
class NG_Unary_Arith_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
      Arguments<(ins NG_TensorType:$arg)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyUnaryArithOp(this); }];
}

// Base class for arithmetic binary operations without side effects.
class NG_Binary_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
      Arguments<(ins NG_TensorType:$lhs, NG_TensorType:$rhs)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
}

// Base class for arithmetic binary operations with verifier.
class NG_Binary_Arith_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, traits>,
      Arguments<(ins NG_TensorType:$lhs, NG_TensorType:$rhs)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyBinaryArithOp(this); }];
}

// Base class for comparison operations with verifier.
class NG_Cmp_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, traits>,
      Arguments<(ins NG_TensorType:$lhs, NG_TensorType:$rhs)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyCmpOp(this); }];
}

// Base class for ternary operations without side effects.
class NG_Ternary_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
      Arguments<(ins NG_TensorType:$op0, NG_TensorType:$op1, NG_TensorType:$op2)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
}


class NG_Axis_Reduction_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
      Arguments<(ins NG_TensorType:$operand, I64ArrayAttr:$axes)>
{
  let summary = "Base class for reduction operations that perform a reduction "
                "across the axes of a  single tensor.";
  let description = [{Axes are represented as an array of I64 attributes.}];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  // TODO
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

// Base class for terminator operations.
class NG_Terminator_Op<string mnemonic, list<OpTrait> traits = []> :
    NG_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<NG_TensorType>:$args)>, Results<(outs)> {}

class NG_Variadic_Result_Op<string mnemonic, list<OpTrait> traits = []> :
    NG_Op<mnemonic, !listconcat(traits, [])>,
    Results<(outs Variadic<NG_TensorType>:$args)> {}

// Terminator Ops
def NGReturnOp : NG_Terminator_Op<"return">;

// ops attributes
include "core/ngraph_dialect/ops_attributes.td"

// Version 0 Ops
include "core/ngraph_dialect/ops_v0.td"

// Version 1 Ops
include "core/ngraph_dialect/ops_v1.td"

// Fused Ops
include "core/ngraph_dialect/fused_ops.td"
