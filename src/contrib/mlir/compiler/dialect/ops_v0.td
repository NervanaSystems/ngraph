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
// This is the nGraph Dialect Version 0 operations definition file.
// All Operations in this file implement OpVersion0 interface.
//===----------------------------------------------------------------------===//

// Unary Operations
def NGAbsOp      : NG_Unary_Arith_Op<"abs",   [OpVersion0]>;
def NGACosOp     : NG_Unary_Arith_Op<"acos",  [OpVersion0]>;
def NGASinOp     : NG_Unary_Arith_Op<"asin",  [OpVersion0]>;
def NGATanOp     : NG_Unary_Arith_Op<"atan",  [OpVersion0]>;
def NGCeilOp     : NG_Unary_Arith_Op<"ceil",  [OpVersion0]>;
def NGConvertOp  : NG_Unary_Arith_Op<"conv",  [OpVersion0]>;
def NGCosOp      : NG_Unary_Arith_Op<"cos",   [OpVersion0]>;
def NGCoshOp     : NG_Unary_Arith_Op<"cosh",  [OpVersion0]>;
def NGExpOp      : NG_Unary_Arith_Op<"exp",   [OpVersion0]>;
def NGFloorOp    : NG_Unary_Arith_Op<"floor", [OpVersion0]>;
def NGNegOp      : NG_Unary_Arith_Op<"neg",   [OpVersion0]>;
def NGLogOp      : NG_Unary_Arith_Op<"log",   [OpVersion0]>;
def NGNotOp      : NG_Unary_Arith_Op<"not",   [OpVersion0]>;
def NGSignOp     : NG_Unary_Arith_Op<"sign",  [OpVersion0]>;
def NGSinOp      : NG_Unary_Arith_Op<"sin",   [OpVersion0]>;
def NGSinhOp     : NG_Unary_Arith_Op<"sinh",  [OpVersion0]>;
def NGTanOp      : NG_Unary_Arith_Op<"tan",   [OpVersion0]>;
def NGTanhOp     : NG_Unary_Arith_Op<"tanh",  [OpVersion0]>;
def NGSqrtOp     : NG_Unary_Arith_Op<"sqrt",  [OpVersion0]>;
def NGReluOp     : NG_Unary_Arith_Op<"relu",  [OpVersion0]>;

// Binary Operations
def NGAddOp      : NG_Binary_Arith_Op<"add", [Commutative, OpVersion0]>;
def NGAndOp      : NG_Binary_Arith_Op<"and", [Commutative, OpVersion0]>;
def NGSubOp      : NG_Binary_Arith_Op<"sub", [OpVersion0]>;
def NGDivOp      : NG_Binary_Arith_Op<"div", [OpVersion0]>;
def NGMaxOp      : NG_Binary_Arith_Op<"max", [Commutative, OpVersion0]>;
def NGMinOp      : NG_Binary_Arith_Op<"min", [Commutative, OpVersion0]>;
def NGMulOp      : NG_Binary_Arith_Op<"mul", [Commutative, OpVersion0]>;
def NGPowOp      : NG_Binary_Arith_Op<"pow", [OpVersion0]>;

// Comparison
def NGEqOp        : NG_Cmp_Op<"equal",      [OpVersion0]>;
def NGGreaterOp   : NG_Cmp_Op<"greater",    [OpVersion0]>;
def NGGreaterEqOp : NG_Cmp_Op<"greater.eq", [OpVersion0]>;
def NGLessOp      : NG_Cmp_Op<"less",       [OpVersion0]>;
def NGLessEqOp    : NG_Cmp_Op<"less.eq",    [OpVersion0]>;
def NGNotEqOp     : NG_Cmp_Op<"not.equal",  [OpVersion0]>;

// Other
def NGSelectOp    : NG_Ternary_Op<"select", [OpVersion0]>
{
  let verifier = [{ return verifyOp(this); }];
}

// Dot Product
def NGDotOp : NG_Binary_Op<"dot", [OpVersion0]>
{
  // TODO: Add reduction axis attribute when needed.
  let verifier = [{ return verifyOp(this); }];
}

// TODO(amprocte): Might be nice to rebase this on some sort of NG_Variadic_Op
// class, but I'm not sure how to add concatenation_axis into the args if we
// do that.
def NGConcatOp :
    NG_OneResult_Op<"concat", [NoSideEffect, OpVersion0]>,
    Arguments<(ins Variadic<NG_TensorType>:$args, I64Attr:$concatenation_axis)>
{
  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];
}

// Axis reduction operations.
def NGSumRedOp : NG_Axis_Reduction_Op<"sum.red", [OpVersion0]>
{
  let summary = "Axis sum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGProdRedOp : NG_Axis_Reduction_Op<"prod.red", [OpVersion0]>
{
  let summary = "Axis product reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGMinRedOp : NG_Axis_Reduction_Op<"min.red", [OpVersion0]>
{
  let summary = "Axis minimum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGMaxRedOp : NG_Axis_Reduction_Op<"max.red", [OpVersion0]>
{
  let summary = "Axis maximum reduction of a tensor.";
  let verifier = [{ return verifyAxisReductionOp(this); }];
}

def NGArgMinRedOp : NG_Axis_Reduction_Op<"argmin.red", [OpVersion0]>
{
  let summary = "Axis minimum index reduction of a tensor.";
  let verifier = [{ return verifyIndexReductionOp(this); }];
}

def NGArgMaxRedOp : NG_Axis_Reduction_Op<"argmax.red", [OpVersion0]>
{
  let summary = "Axis maximum index reduction of a tensor.";
  let verifier = [{ return verifyIndexReductionOp(this); }];
}

def NGAllRedOp : NG_Axis_Reduction_Op<"all.red", [OpVersion0]>
{
  let summary = "Axis logical AND reduction of a boolean tensor.";
  let verifier = [{ return verifyLogicalReductionOp(this); }];
}

def NGAnyRedOp : NG_Axis_Reduction_Op<"any.red", [OpVersion0]>
{
  let summary = "Axis logical OR reduction of a boolean tensor.";
  let verifier = [{ return verifyLogicalReductionOp(this); }];
}

// Gather
def NGGatherOp : 
    NG_OneResult_Op<"gather", [NoSideEffect, OpVersion0]>,
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
    NG_OneResult_Op<"convolution", [NoSideEffect, OpVersion0]>,
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
    unsigned getVersion() { return 0; }
  }];
}

