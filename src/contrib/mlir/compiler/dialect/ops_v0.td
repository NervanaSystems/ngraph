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
  }];
}

// AvgPool
def NGAvgPoolOp :
    NG_OneResult_Op<"avgPool", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType:$arg, 
               I64ArrayAttr     :$windowShape,
               I64ArrayAttr     :$windowMovementStrides,
               I64ArrayAttr     :$padBelow,
               I64ArrayAttr     :$padAbove,
               BoolAttr         :$includePadding,
               PadTypeEnumAttr  :$padType,
               BoolAttr         :$ceilMode
               )>
{
    let summary = "Batched average pooling operation, with optional padding and window stride."
    let description = [{
      Constructs a batched average pooling operation.
      
      arg                   The output producing the input data batch tensor
      windowShape           The window shape
      windowMovementStrides The window movement strides
      paddingBelow         The below-padding shape
      paddingAbove         The above-padding shape
      
      includePadding        If true then averages include padding elements, 
                            each treated as the number zero.  If false, padding elements are
                            entirely ignored when computing averages
      padType               Padding type to use for additional padded dimensions 
      ceilMode              Whether to use ceiling while computing output shape
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res, Value *arg, 
       ArrayAttr windowShape, ArrayAttr windowMovementStrides, 
       ArrayAttr padBelow, ArrayAttr padAbove, BoolAttr includPadding, IntegerAttr padType",
      [{
        tblgen_state.addOperands(arg);
        tblgen_state.addAttribute("windowShape", windowShape);
        tblgen_state.addAttribute("windowMovementStrides", windowMovementStrides);
        tblgen_state.addAttribute("padBelow", padBelow);
        tblgen_state.addAttribute("padAbove", padAbove);
        tblgen_state.addAttribute("includPadding", includPadding);
        tblgen_state.addAttribute("padType", padType);
        tblgen_state.addAttribute("ceilMode", false);
        tblgen_state.addTypes(res);
      }]>,

      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res, Value *arg, 
       ArrayAttr windowShape, ArrayAttr windowMovementStrides, 
       ArrayAttr padBelow, ArrayAttr padAbove, BoolAttr includPadding",
      [{
        tblgen_state.addOperands(arg);
        tblgen_state.addAttribute("windowShape", windowShape);
        tblgen_state.addAttribute("windowMovementStrides", windowMovementStrides);
        tblgen_state.addAttribute("padBelow", padBelow);
        tblgen_state.addAttribute("padAbove", padAbove);
        tblgen_state.addAttribute("includPadding", includPadding);
        tblgen_state.addAttribute("padType", mlir::PadType::EXPLICIT);
        tblgen_state.addAttribute("ceilMode", false);
        tblgen_state.addTypes(res);
      }]>
    ];

    let extraClassDeclaration = [{
      void setWindowShape(const ArrayAttr& arrayAttr)           { this->setAttr("windowShape", arrayAttr);          }
      void setWindowMovementStrides(const ArrayAttr& arrayAttr) { this->setAttr("windowMovementStrides", arrayAttr);}
      void setPadBelow(const ArrayAttr& arrayAttr)      { this->setAttr("padBelow", arrayAttr); }
      void setPadAbove(const ArrayAttr& arrayAttr)      { this->setAttr("padAbove", arrayAttr); }    
      void setIncludePadding(const BoolAttr& boolAttr)  { this->setAttr("includePadding", boolAttr); }
      void setPadType(const IntegerAttr& intAttr)       { this->setAttr("padType", intAttr);         }
      void setCeilMode(const BoolAttr& boolAttr)        { this->setAttr("ceilMode", boolAttr);       }
    }];
}

// AvgPool for back prop
def NGAvgPoolBackPropOp :
    NG_OneResult_Op<"avgPoolBackProp", [NoSideEffect, OpVersion0]>,
    Arguments<(ins I64ArrayAttr :$forwardArgShape,
               NG_TensorType    :$delta, 
               I64ArrayAttr     :$windowShape,
               I64ArrayAttr     :$windowMovementStrides,
               I64ArrayAttr     :$padBelow,
               I64ArrayAttr     :$padAbove,
               BoolAttr         :$includePadding
               )>
{
    let summary = "Batched backprop average pooling operation, with optional padding and window stride."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    
    let extraClassDeclaration = [{
      void setForwardArgShape(const ArrayAttr& arrayAttr) { this->setAttr("forwardArgShape", arrayAttr); }
      void setWindowShape(const ArrayAttr& arrayAttr)     { this->setAttr("windowShape", arrayAttr);     }
      void setWindowMovementStrides(const ArrayAttr& arrayAttr) { this->setAttr("windowMovementStrides", arrayAttr);}
      void setPadBelow(const ArrayAttr& arrayAttr)      { this->setAttr("padBelow", arrayAttr); }
      void setPadAbove(const ArrayAttr& arrayAttr)      { this->setAttr("padAbove", arrayAttr); }    
      void setIncludePadding(const BoolAttr& boolAttr)  { this->setAttr("includePadding", boolAttr); }
    }];
}

// BatchNorm for Training
def NGBacthNormTrainingOp :
    NG_OneResult_Op<"batchNormTraining", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType :$input, 
               NG_TensorType     :$gamma, 
               NG_TensorType     :$beta, 
               F64Attr           :$epsilon
               )>
{
    let summary = "BatchNorm for training."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];

    let extraClassDeclaration = [{
      void setEpsilon(const Attr& attr)  { this->setAttr("epsilon", attr);  }
    }];
}

// BatchNorm for Inference
def NGBacthNormInferenceOp :
    NG_OneResult_Op<"batchNormInference", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType :$input, 
               NG_TensorType     :$gamma, 
               NG_TensorType     :$beta, 
               NG_TensorType     :$mean, 
               NG_TensorType     :$variance, 
               F64Attr           :$epsilon
               )>
{
    let summary = "BatchNorm for training."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    let extraClassDeclaration = [{
      void setEpsilon(const Attr& attr)  { this->setAttr("epsilon", attr);  }
    }];
}


// BatchNorm for Training BackProp
def NGBacthNormTrainingBackPropOp :
    NG_OneResult_Op<"batchNormTrainingBackProp", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType :$input, 
               NG_TensorType     :$gamma, 
               NG_TensorType     :$beta, 
               NG_TensorType     :$mean, 
               NG_TensorType     :$variance, 
               NG_TensorType     :$delta, 
               F64Attr           :$epsilon
               )>
{
    let summary = "BatchNorm for training."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    let extraClassDeclaration = [{
      void setEpsilon(const Attr& attr)  { this->setAttr("epsilon", attr);  }
    }];
}

// Broadcast
def NGBroadcastOp :
    NG_OneResult_Op<"broadcast", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType :$arg, 
               I64ArrayAttr      :$shape,
               I64ArrayAttr      :%axisSet
               )>
{
    let summary = "Operation which "adds" axes to an input tensor, replicating elements from the
                   input as needed along the new axes."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    let extraClassDeclaration = [{
      void setAxisSet(const ArrayAttr& attr)  { this->setAttr("axisSet", attr); }
      void setShape(const ArrayAttr& attr)    { this->setAttr("shape", attr);   }
    }];
}

// Broadcast distributed
def NGBroadcastOp :
    NG_OneResult_Op<"broadcast", [NoSideEffect, OpVersion0]>,
    Arguments<(ins NG_TensorType :$arg, 
               I64ArrayAttr      :$shape,
               I64ArrayAttr      :%axisSet
               )>
{
    let summary = "Operation which "adds" axes to an input tensor, replicating elements from the
                   input as needed along the new axes."
    let description = [{
      // TBD
    }]
    
    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];
    let extraClassDeclaration = [{
      void setAxisSet(const ArrayAttr& attr)  { this->setAttr("axisSet", attr); }
      void setShape(const ArrayAttr& attr)    { this->setAttr("shape", attr);   }
    }];
}



