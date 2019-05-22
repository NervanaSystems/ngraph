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
// This is the NGraph Dialect operation definition file.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

// NGraph Dialect operations definitions
//
// This files declares NGraph operations that table-gen uses to create C++ code
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


// NGraph Types
// This defines records equivalent to NGraph types. It doesn't generate code.
// This is used as a type in the DAG input/outputs.
// Constraints (CPred) are used to type-check args/results of that type during op verification
def NG_TensorType : Type<CPred<"{0}.isa<ngraph::runtime::ngmlir::NGTensorType>()">,
                     "NGraph Tensor Type">;

// A generic un-typed MemRef. Used for Fake instructions inserted during dialect lowering
def NG_MemRefType : Type<IsMemRefTypePred, "MemRef Type">;

// NGraph operation base class.
// Prepends "ng." to operation name
class NG_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<!strconcat("ng.", mnemonic), traits> {}

// Operations producing single result.
// Will set OneResult trait based on Results out dag.
class NG_OneResult_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs NG_TensorType:$res)> {}

// Base for fake instructions defining MemRef values
class NG_MemRefDef_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs NG_MemRefType:$res)> {}

// Operations producing no results
class NG_ZeroResult_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs)> {}

// Arithmetic binary operations
// Input and outputs have same type
class NG_Unary_Arith_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect, SameValueType], traits)>,
      Arguments<(ins NG_TensorType:$arg)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_FAIL() << "No parser support"; return false; }];

  let verifier = [{ return verifyUnaryArithOp(this); }];
}

// Arithmetic binary operations
// Inputs and outputs have same type
class NG_Binary_Arith_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_OneResult_Op<mnemonic, !listconcat([NoSideEffect, SameValueType], traits)>,
      Arguments<(ins NG_TensorType:$lhs, NG_TensorType:$rhs)>
{
  // TODO: Implement
  let parser = [{ NGRAPH_FAIL() << "No parser support"; return false; }];

  let verifier = [{ return verifyBinaryArithOp(this); }];
}

// Base class for terminator operations.
class NG_Terminator_Op<string mnemonic, list<OpTrait> traits = []> :
    NG_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<NG_TensorType>:$args)>, Results<(outs)> {}

// Unary Operations
def NGAbsOp      : NG_Unary_Arith_Op<"abs">;
def NGACosOp     : NG_Unary_Arith_Op<"acos">;
def NGASinOp     : NG_Unary_Arith_Op<"asin">;
def NGATanOp     : NG_Unary_Arith_Op<"atan">;
def NGCeilOp     : NG_Unary_Arith_Op<"ceil">;
def NGConvertOp  : NG_Unary_Arith_Op<"conv">;
def NGCosOp      : NG_Unary_Arith_Op<"cos">;
def NGCoshOp     : NG_Unary_Arith_Op<"cosh">;
def NGExpOp      : NG_Unary_Arith_Op<"exp">;
def NGFloorOp    : NG_Unary_Arith_Op<"floor">;
def NGLogOp      : NG_Unary_Arith_Op<"log">;
def NGNegOp      : NG_Unary_Arith_Op<"neg">;
def NGNotOp      : NG_Unary_Arith_Op<"not">;
def NGSignOp     : NG_Unary_Arith_Op<"sign">;
def NGSinOp      : NG_Unary_Arith_Op<"sin">;
def NGSinhOp     : NG_Unary_Arith_Op<"sinh">;
def NGTanOp      : NG_Unary_Arith_Op<"tan">;
def NGTanhOp     : NG_Unary_Arith_Op<"tanh">;
def NGSqrtOp     : NG_Unary_Arith_Op<"sqrt">;

// Binary Operations
def NGAddOp : NG_Binary_Arith_Op<"add", [Commutative]>;
def NGAndOp : NG_Binary_Arith_Op<"and", [Commutative]>;
def NGSubOp : NG_Binary_Arith_Op<"sub">;
def NGDivOp : NG_Binary_Arith_Op<"div">;
def NGMaxOp : NG_Binary_Arith_Op<"max", [Commutative]>;
def NGMinOp : NG_Binary_Arith_Op<"min", [Commutative]>;
def NGMulOp : NG_Binary_Arith_Op<"mul", [Commutative]>;
def NGPowOp : NG_Binary_Arith_Op<"pow">;

// Comparison
def NGEqOp        : NG_OneResult_Op<"equal",      [NoSideEffect]>;
def NGGreaterOp   : NG_OneResult_Op<"greater",    [NoSideEffect]>;
def NGGreaterEqOp : NG_OneResult_Op<"greater.eq", [NoSideEffect]>;
def NGLessOp      : NG_OneResult_Op<"less",       [NoSideEffect]>;
def NGLessEqOp    : NG_OneResult_Op<"less.eq",    [NoSideEffect]>;
def NGNotEqOp     : NG_OneResult_Op<"not.equal",  [NoSideEffect]>;

// Other
def NGSelectOp    : NG_OneResult_Op<"select", [NoSideEffect]>;

// Matrix Multiply
def NGMatMulBiasOp : NG_Binary_Arith_Op<"matmul.bias">
{
  let verifier=[{return verifyOp(this);}];
}

// Terminator Ops
def NGReturnOp : NG_Terminator_Op<"return">;

// Fake ops
def NGFakeInputOp : NG_MemRefDef_Op<"fake.input", [NoSideEffect]>;