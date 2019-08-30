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

// Base for fake instructions defining MemRef values
class NG_MemRefDef_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Op<mnemonic, traits>, Results<(outs NG_MemRefType:$res)> {}

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
def NGReluOp     : NG_Unary_Arith_Op<"relu">;

// Binary Operations
def NGAddOp      : NG_Binary_Arith_Op<"add", [Commutative]>;
def NGAndOp      : NG_Binary_Arith_Op<"and", [Commutative]>;
def NGSubOp      : NG_Binary_Arith_Op<"sub">;
def NGDivOp      : NG_Binary_Arith_Op<"div">;
def NGMaxOp      : NG_Binary_Arith_Op<"max", [Commutative]>;
def NGMinOp      : NG_Binary_Arith_Op<"min", [Commutative]>;
def NGMulOp      : NG_Binary_Arith_Op<"mul", [Commutative]>;
def NGPowOp      : NG_Binary_Arith_Op<"pow">;

// Comparison
def NGEqOp        : NG_Cmp_Op<"equal">;
def NGGreaterOp   : NG_Cmp_Op<"greater">;
def NGGreaterEqOp : NG_Cmp_Op<"greater.eq">;
def NGLessOp      : NG_Cmp_Op<"less">;
def NGLessEqOp    : NG_Cmp_Op<"less.eq">;
def NGNotEqOp     : NG_Cmp_Op<"not.equal">;

// Other
def NGSelectOp    : NG_Ternary_Op<"select">
{
  let verifier = [{ return verifyOp(this); }];
}

// Dot Product
def NGDotOp : NG_Binary_Op<"dot">
{
  // TODO: Add reduction axis attribute when needed.
  let verifier = [{ return verifyOp(this); }];
}

// TODO(amprocte): Might be nice to rebase this on some sort of NG_Variadic_Op
// class, but I'm not sure how to add concatenation_axis into the args if we
// do that.
def NGConcatOp :
    NG_OneResult_Op<"concat", [NoSideEffect]>,
    Arguments<(ins Variadic<NG_TensorType>:$args, I64Attr:$concatenation_axis)>
{
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];
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

// Axis reduction operations.
def NGSumRedOp : NG_Axis_Reduction_Op<"sum.red">
{
  let summary = "Axis sum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGProdRedOp : NG_Axis_Reduction_Op<"prod.red">
{
  let summary = "Axis product reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGMinRedOp : NG_Axis_Reduction_Op<"min.red">
{
  let summary = "Axis minimum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGMaxRedOp : NG_Axis_Reduction_Op<"max.red">
{
  let summary = "Axis maximum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGArgMinRedOp : NG_Axis_Reduction_Op<"argmin.red">
{
  let summary = "Axis minimum index reduction of a tensor.";
  let verifier = [{ return verifyIndexReductionOp(this); }];
}

def NGArgMaxRedOp : NG_Axis_Reduction_Op<"argmax.red">
{
  let summary = "Axis maximum index reduction of a tensor.";
  let verifier = [{ return verifyIndexReductionOp(this); }];
}

def NGAllRedOp : NG_Axis_Reduction_Op<"all.red">
{
  let summary = "Axis logical AND reduction of a boolean tensor.";
  let verifier = [{ return verifyLogicalReductionOp(this); }];
}

def NGAnyRedOp : NG_Axis_Reduction_Op<"any.red">
{
  let summary = "Axis logical OR reduction of a boolean tensor.";
  let verifier = [{ return verifyLogicalReductionOp(this); }];
}

// Gather
def NGGatherOp : 
    NG_OneResult_Op<"gather", [NoSideEffect]>,
    Arguments<(ins NG_TensorType:$params, NG_TensorType:$indices, I64Attr:$axis)>
{
  let summary = "Gather slices from params along the specified axis according to indices";
  let description = [{
    Gather slices from axis of params according to indices
    params The tensor from which slices are gathered
    indices Index tensor. Data type must be `element::i32` or `element::i64`
    axis Axis in params to gather
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];
}

// Convolution
def NGConvolutionOp :
    NG_OneResult_Op<"convolution", [NoSideEffect]>,
    Arguments<(ins NG_TensorType:$images, NG_TensorType:$filters, 
               I64ArrayAttr:$strides,
               I64ArrayAttr:$padBelow,
               I64ArrayAttr:$padAbove)>
{
  let summary = "Convolution of a tensor of filters over a tensor of images with padding support";
  let description = [{
    Convolution operation with padding and stride support. No dilation supported.
    images        Input image tensor. Shape is [N, C_IN, D1, ... Df]
    filters       Set of filters to apply. Shape is [C_OUT, C_IN, F1, ... Ff]
    strides       Window movement strides. Shape is [f]. Attribute.
    padBelow      The padding-below sizes. Shape is [f]. Attribute.
    padAbove      The padding-below sizes. Shape is [f]. Attribute.
   
    Output is of shape [N, C_OUT, R1, ... Rf]
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return verifyOp(this); }];
  let extraClassDeclaration = [{
    void setStrides(ArrayAttr& arrayAttr)  { this->setAttr("strides", arrayAttr);  }
    void setPadBelow(ArrayAttr& arrayAttr) { this->setAttr("padBelow", arrayAttr); }
    void setPadAbove(ArrayAttr& arrayAttr) { this->setAttr("padAbove", arrayAttr); }
  }];
}

// Terminator Ops
def NGReturnOp : NG_Terminator_Op<"return">;

// Fake ops
def NGFakeInputOp : NG_MemRefDef_Op<"fake.input", [NoSideEffect]>;
