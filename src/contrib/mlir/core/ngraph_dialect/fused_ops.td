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
// This is the nGraph Dialect Fused Ops definition file
// All Operations in this file implement FusedOp interface.
//===----------------------------------------------------------------------===//

#ifdef NG_FUSED_OPS
#else
#define NG_FUSED_OPS
// Squeeze Op
def NGSqueezeOp : 
    NG_OneResult_Op<"squeeze", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$axes)>
{
  let summary = "Squeeze Op";
  let description = [{
    Squeeze Op
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];

}

// Unsqueeze Op
def NGUnSqueezeOp : 
    NG_OneResult_Op<"unsqueeze", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$axes)>
{
  let summary = "Unsqueeze Op";
  let description = [{
    Unsqueeze Op
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];

}

// Squared Difference Op
def NGSquaredDiffOp : 
    NG_Binary_Op<"sqrtdiff", [DeclareOpInterfaceMethods<FusedOp>]>
{
  let summary = "Squared Difference Op";
  let description = [{
    Squared Difference Op
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];

}

// Split Op
def NGSplitOp : 
    NG_Variadic_Result_Op<"split", [DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$axis, I64ArrayAttr:$numSplits)>
{
  let summary = "Split op";
  let description = [{ 
    Splits the input tensor into a list of smaller tensors ("pieces")
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];

  let extraClassDeclaration = [{
    void setAxis(const Attribute& attr)                 { this->setAttr("axis", attr);            }
    void setNumSplits(const ArrayAttr& arrayAttr)  { this->setAttr("numSplits", arrayAttr);  }
  }];
  
}

// SpaceToDepth Op
def NGSpaceToDepthOp :
    NG_OneResult_Op<"spaceToDepth", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, I64Attr:$blockSize)>
{
    let summary = "Space to depth op";
    let description = [{
      SpaceToDepth permutes input tensor blocks of spatial data into depth dimension.
      Values from the height and width dimensions are moved to the depth dimension.
      Output node produces a tensor with shape:
        [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

    let verifier = [{ return verifyOp(this); }];

    let extraClassDeclaration = [{
      void setBlockSize(const Attribute& attr) { this->setAttr("blockSize", attr); }
    }];

}
    
// ShuffleChannels Op
def NGShuffleChannelsOp :
    NG_OneResult_Op<"shuffleChannels", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, I64Attr:$axis, I64Attr:$groups)>
{
    let summary = "Shuffle Channels op";
    let description = [{
      Constructs a ShuffleChannels node.
      data - Node producing the input tensor
      axis - channel dimension index in the data tensor. A negative value means
             that the index should be calculated from the back of the input data
             shape.
      groups - number of groups the channel dimension specified by axis should be
             split into
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

    let verifier = [{ return verifyOp(this); }];

    let extraClassDeclaration = [{
      void setAxis(const Attribute& axis)     { this->setAttr("axis", axis); }
      void setGroups(const Attribute& groups) { this->setAttr("groups", groups); }
    }];
}

// ScaleShift Op
def NGScaleShiftOp :
    NG_Ternary_Op<"scaleShift", [DeclareOpInterfaceMethods<FusedOp>]>
{
    let summary = "scaleShift op";
    let description = [{
      Operator performing Scale Shift transformation.
      Y = Scale * Data + Shift
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

    let verifier = [{ return verifyOp(this); }];
}

// RNN Cell Op
def NGRNNCellOp : 
    NG_OneResult_Op<"rnnCell", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$W, NG_TensorType:$R, NG_TensorType:$H_t, I64Attr:$hiddenSize)>
{
    let summary = "RNN Cell";
    let description = [{
      RNN Cell
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

    let verifier = [{ return verifyOp(this); }];  

    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *X, Value* W, Value* R, Value* H_t, "
      "Attribute hiddenSize, ArrayAttr activations,"
      "ArrayAttr activationAlpha, ArrayAttr activationBeta, Attribute clip", [{
         tblgen_state.addOperands(X);
         tblgen_state.addOperands(W);
         tblgen_state.addOperands(R);
         tblgen_state.addOperands(H_t);
         tblgen_state.addAttribute("hiddenSize", hiddenSize);
         tblgen_state.addAttribute("activations", activations);
         tblgen_state.addAttribute("activationsAlpha", activationsAlpha);
         tblgen_state.addAttribute("activationsBeta", activationsBeta);
         tblgen_state.addAttribute("clip", clip);
         tblgen_state.addTypes(res);
       }]>
    ];

    let extraClassDeclaration = [{
      void setHiddenSize(const Attribute& attr)       { this->setAttr("hiddenSize", attr);  }
      void setActivations(const ArrayAttr& attr)      { this->setAttr("activations", attr); }
      void setActivationsAlpha(const ArrayAttr& attr) { this->setAttr("activationsAlpha", attr); }
      void setActivationsBeta(const ArrayAttr& attr)  { this->setAttr("activationsBeta", attr);  }
      void setClip(const Attribute& attr)             { this->setAttr("clip", attr); }
    }];
}

// Prelu Op
def NGPrelu : 
    NG_OneResult_Op<"prelu", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$slope)>
{
    let summary = "Prelu op";
    let description = [{
      Prametrized Relu
      x <  0 => f(x) = x * slope
      x >= 0 => f(x) = x
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];  
}

// Normalize L2 Op
def NGNormalizeL2Op : 
    NG_OneResult_Op<"normalizeL2", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$axis, F32Attr:$eps, EpsModeEnumAttr:$epsMode)>
{
    let summary = "NormalizeL2 op";
    let description = [{
      Constructs a Normalize operation.
      data            - Node producing the input tensor
      axes            - Node indicating axes along which reduction is
                        calculated
      eps             - The epsilon added to L2 norm.
      eps_mode        - Specifies how eps is combined with L2 value calculated
                        before division
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];  

    let extraClassDeclaration = [{
      void setEpsMode(const Attribute& epsMode) { this->setAttr("epsMOde", epsMode); }
      void setEps(const Attribute& eps) { this->setAttr("eps", eps); }
    }];
}

// MVN Op
def NGMVN : 
    NG_OneResult_Op<"mvn", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, 
                   DefaultValuedAttr<BoolAttr, "true">: $acrossChannels,
                   DefaultValuedAttr<BoolAttr, "true">: $normalizeVariance,
                   DefaultValuedAttr<F32Attr, "1e-9"> : $eps)>
{
    let summary = "MVN op";
    let description = [{
      data Input tensor with data
      normalize_variance flag that denotes whether to perform variance
                         normalization.
      across_channels    flag that denotes if mean values are shared across channels.
      eps the number     to be added to the variance to avoid division by zero when
                         normalizing the value
      reduction_axes     a list of axes, along which to reduce.
    }];

    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *data, ArrayAttr reductionAxes, Attribute normalizeVariance,"
      "Attribute eps", [{
         tblgen_state.addOperands(data);
         tblgen_state.addAttribute("reductionAxes", reductionAxes);
         tblgen_state.addAttribute("normalizeVariance", normalizeVariance);
         tblgen_state.addAttribute("eps", eps);
         tblgen_state.addTypes(res);
       }]>
    ];

    let extraClassDeclaration = [{
      void setAcrossChannels(const Attribute& attr)     { this->setAttr("acrossChannels", attr); }
      void setNormalizeVariance(const Attribute& attr)  { this->setAttr("normalizeVariance", attr); }
      void setEps(const Attribute& attr)                { this->setAttr("eps", attr); }
      void setReductionAxes(const ArrayAttr& attr)      { this->setAttr("reductionAxes", attr); }
    }];
}

// MatMul Op
def NGMatMul : 
  NG_OneResult_Op<"matmul", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
  Arguments<(ins NG_TensorType:$A, NG_TensorType:$B, 
            DefaultValuedAttr<BoolAttr, "false">:$transposeA,
            DefaultValuedAttr<BoolAttr, "false">:$transposeB)>
{
    let summary = "MatMul op";
    let description = [{
      A Matrix A
      B Matrix B
      transpose_a If matrix A should be transposed.
      transpose_b If matrix B should be transposed.
    }];

    let extraClassDeclaration = [{
      void setTransposeA(const Attribute& attr) { this->setAttr("transposeA", attr); }
      void setTransposeB(const Attribute& attr) { this->setAttr("transposeB", attr); }
    }];
}

// LSTM Cell Op
def NGLSTMCellOp : 
    NG_OneResult_Op<"lstmCell", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$W, NG_TensorType:$R,
               NG_TensorType:$H_t, NG_TensorType:$C_t,
               NG_TensorType:$B, NG_TensorType:$P,
               I64Attr:$hiddenSize,
               DefaultValuedAttr<StrArrayAttr, "\"sigmoid\",\"tanh\",\"tanh\"">:$activations,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationAlpha,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationBeta,
               DefaultValuedAttr<F32Attr, "0.0">:$clip,
               DefaultValuedAttr<BoolAttr, "false">:$inputForget)>
{
    let summary = "LSTM Cell";
    let description = [{
      LSTM Cell
    }];

    let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
    let verifier = [{ return verifyOp(this); }];  


    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *X, Value* W, Value* R, Value* H_t, Value* C_t,"
      "Attribute hiddenSize, ArrayAttr activations,"
      "ArrayAttr activationAlpha, ArrayAttr activationBeta,"
      "Attribute clip, Attribute inputForget", [{
         tblgen_state.addOperands({X, W, R, H_t, C_t});
         tblgen_state.addAttribute("hiddenSize", hiddenSize);
         tblgen_state.addAttribute("activations", activations);
         tblgen_state.addAttribute("activationsAlpha", activationsAlpha);
         tblgen_state.addAttribute("activationsBeta", activationsBeta);
         tblgen_state.addAttribute("clip", clip);
         tblgen_state.addAttribute("inputForget", inputForget);
         tblgen_state.addTypes(res);
      }]>,

       OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *X, Value* W, Value* R, Value* H_t, Value* C_t,"
      "Attribute hiddenSize",
      [{
         tblgen_state.addOperands({X, W, R, H_t, C_t});
         tblgen_state.addAttribute("hiddenSize", hiddenSize);
         tblgen_state.addTypes(res);
      }]>
    ];

    let extraClassDeclaration = [{
      void setHiddenSize  (const Attribute& attr)       { this->setAttr("hiddenSize", attr); }
      void setActivations (const ArrayAttr& attr)       { this->setAttr("activation", attr); }
      void setActivationsAlpha (const ArrayAttr& attr)  { this->setAttr("activationsAlpha", attr); }
      void setActivationsBeta (const ArrayAttr& attr)   { this->setAttr("activationsBeta", attr); }
      void setClip(const Attribute& attr)               { this->setAttr("clip", clip); }

    }];
}

            LayerNorm(const Output<Node>& data,
                      const Output<Node>& scale,
                      const Output<Node>& bias,
                      bool keep_stats = true,
                      int64_t begin_norm_axis = 1,
                      double epsilon = 1e-5);

            LayerNorm(const Output<Node>& data,
                      bool keep_stats = true,
                      int64_t begin_norm_axis = 1,
                      double epsilon = 1e-5);

// LayerNorm Op
def NGLayerNormOp : 
    NG_OneResult_Op<"layernorm", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>],
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$scale, NG_TensorType:$bias,
               DefaultValuedAttr<BoolAttr, "true">:$keepStats,
               DefaultValuedAttr<I64Attr, "1">:$beginNormAxis,
               DefaultValuedAttr<F64Attr, "1e-5">:$epsilon)>
{
    let summary = "LayerNorm Op";
    let description = "Constructs an LayerNorm operation."
    
    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *data, Attribute keepStats, Attribute beginNormAxis, Attribute epsilon", [{
         tblgen_state.addOperands(data);
         tblgen_state.addAttribute("keepStats", keepStats);
         tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
         tblgen_state.addAttribute("epsilon", epsilon);
         tblgen_state.addTypes(res);
      }]>
    ];

    let extraClassDeclaration = [{
      void setKeepstats(const Attribute& attr)      { this->setAttr("keepStats", attr);    }
      void setBeginNormAxis(const Attribute& attr)  { this->setAttr("beginNormAxis", attr);}
      void setEpsilonAxis(const Attribute& attr)    { this->setAttr("epsilon", attr);      }
    }]

}


// LayerNormBackProp Op
def NGLayerNormBackPropOp : 
    NG_OneResult_Op<"layernorm", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>],
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$delta, 
               NG_TensorType:$mean, NG_TensorType:$variance, NG_TensorType:$scale,
               DefaultValuedAttr<I64Attr, "1">:$beginNormAxis,
               DefaultValuedAttr<F64Attr, "1e-5">:$epsilon)>
{
    let summary = "LayerNorm Op";
    let description = "Constructs an LayerNorm operation."
    
    let builders = [
      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *data, Value *delta, Value *mean, Value *variance,"
      "Attribute beginNormAxis, Attribute epsilon", [{
         tblgen_state.addOperands({data, delta, mean, variance});
         tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
         tblgen_state.addAttribute("epsilon", epsilon);
         tblgen_state.addTypes(res);
      }]>,

      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *data, Value *delta, Value *scale,"
      "Attribute beginNormAxis, Attribute epsilon", [{
         tblgen_state.addOperands({data, delta, scale});
         tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
         tblgen_state.addAttribute("epsilon", epsilon);
         tblgen_state.addTypes(res);
      }]>,

      OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *data, Value *delta,"
      "Attribute beginNormAxis, Attribute epsilon", [{
         tblgen_state.addOperands({data, delta});
         tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
         tblgen_state.addAttribute("epsilon", epsilon);
         tblgen_state.addTypes(res);
      }]>,
    ];

    let extraClassDeclaration = [{
      void setBeginNormAxis(const Attribute& attr)  { this->setAttr("beginNormAxis", attr);}
      void setEpsilonAxis(const Attribute& attr)    { this->setAttr("epsilon", attr);      }
    }]
}

// HardSigmoid Op
def NGHardSigmoid : 
    NG_OneResult_Op<"hardsigmoid", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>],
    Arguments<(ins NG_TensorType:$data,
               F64Attr:$alpha, F64Attr:$beta)>
{
    let summary = "Hard sigmoid op";
    let description = [{
      Parameterized, bounded sigmoid-like, piecewise linear
      function. min(max(alpha*x + beta, 0), 1)
    }];

    let extraClassDeclaration = [{
      void setAlpha(const Attribute& attr) { this->setAttr("alpha", attr); }
      void setBeta(const Attribute& attr)  { this->setAttr("beta", attr); }
    }];
}
    
#endif //NG_FUSED_OPS
