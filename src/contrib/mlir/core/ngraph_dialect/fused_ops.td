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
  let verifier = [{ return mlir::success(); /* TBD */ }];
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
  let verifier = [{ return mlir::success(); /* TBD */ }];
}

// Squared Difference Op
def NGSquaredDiffOp : 
    NG_Binary_Op<"sqrdDiff", [DeclareOpInterfaceMethods<FusedOp>]>
{
  let summary = "Squared Difference Op";
  let description = [{
    Squared Difference Op
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];
}

// Split Op
def NGSplitOp : 
    NG_Variadic_Result_Op<"split", [DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, I64Attr:$axis, I64ArrayAttr:$numSplits)>
{
  let summary = "Split op";
  let description = [{ 
    Splits the input tensor into a list of smaller tensors ("pieces")
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];

  let extraClassDeclaration = [{
    void setAxis(const Attribute& attr)            { this->setAttr("axis", attr);            }
    void setNumSplits(const ArrayAttr& arrayAttr)  { this->setAttr("numSplits", arrayAttr);  }
  }];
 
}

// SpaceToDepth Op
def NGSpaceToDepthOp :
    NG_OneResult_Op<"spaceToDepth", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, 
              DefaultValuedAttr<I64Attr, "1">:$blockSize,
              DepthSpaceModeEnumAttr:$mode)>
{
  let summary = "Space to depth op";
  let description = [{
    SpaceToDepth permutes input tensor blocks of spatial data into depth dimension.
    Values from the height and width dimensions are moved to the depth dimension.
    Output node produces a tensor with shape:
      [N, C * blocksize * blocksize, H / blocksize, W / blocksize]

    data       Node producing the input tensor
    blockSize  The size of the block of values to be moved
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];

  let extraClassDeclaration = [{
    void setBlockSize(const Attribute& attr) { this->setAttr("blockSize", attr); }
    void setMode(const Attribute& attr)      { this->setAttr("mode", attr);      } 
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
    data    Node producing the input tensor
    axis    channel dimension index in the data tensor. A negative value means
            that the index should be calculated from the back of the input data
            shape.
    groups  number of groups the channel dimension specified by axis should be
            split into
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];

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
  let verifier = [{ return mlir::success(); /* TBD */ }];
}

// RNN Cell Op
def NGRNNCellOp : 
    NG_OneResult_Op<"rnnCell", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$W, NG_TensorType:$R, NG_TensorType:$H_t,
               Variadic<NG_TensorType>:$optionalArgs, 
               DefaultValuedAttr<StrArrayAttr, "{\"tanh\"}">:$activations,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationAlpha,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationBeta,
               DefaultValuedAttr<F32Attr, "0.0">:$clip,
               I64Attr:$hiddenSize)>
{
  let summary = "RNN Cell";
  let description = [{
    RNN Cell
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];  

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &tblgen_state, Type res,"
    "Value *X, Value* W, Value* R, Value* H_t, "
    "Attribute hiddenSize, ArrayAttr activations,"
    "ArrayAttr activationAlpha, ArrayAttr activationBeta, Attribute clip", [{
      tblgen_state.addOperands({X, W, R, H_t});
      tblgen_state.addAttribute("hiddenSize", hiddenSize);
      tblgen_state.addAttribute("activations", activations);
      tblgen_state.addAttribute("activationAlpha", activationAlpha);
      tblgen_state.addAttribute("activationBeta", activationBeta);
      tblgen_state.addAttribute("clip", clip);
      tblgen_state.addTypes(res);
    }]>
  ];

  let extraClassDeclaration = [{
    void setHiddenSize(const Attribute& attr)       { this->setAttr("hiddenSize", attr);  }
    void setActivations(const ArrayAttr& attr)      { this->setAttr("activations", attr); }
    void setActivationAlpha(const ArrayAttr& attr)  { this->setAttr("activationAlpha", attr); }
    void setActivationBeta(const ArrayAttr& attr)   { this->setAttr("activationBeta", attr);  }
    void setClip(const Attribute& attr)             { this->setAttr("clip", attr); }

    // get bias operand if present
    Value* B() 
    { 
      auto varArgs = optionalArgs();
      return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
    }
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
  let verifier = [{ return mlir::success(); /* TBD */ }];  
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
  let verifier = [{ return mlir::success(); /* TBD */ }];  

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
// 
def NGLSTMCellOp : 
    NG_OneResult_Op<"lstmCell", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$W, NG_TensorType:$R,
               NG_TensorType:$H_t, NG_TensorType:$C_t,
               Variadic<NG_TensorType>:$optionalArgs, 
               I64Attr:$hiddenSize,
               DefaultValuedAttr<LSTMWeightsFormatEnumAttr, 
                                 "static_cast<int64_t>(MLIRLSTMWeightsFormat::IFCO)">:$weightFormat,
               DefaultValuedAttr<StrArrayAttr, "{\"sigmoid\",\"tanh\",\"tanh\"}">:$activations,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationAlpha,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationBeta,
               DefaultValuedAttr<F32Attr, "0.0">:$clip,
               DefaultValuedAttr<BoolAttr, "false">:$inputForget)>
{
  let summary = "LSTM Cell";
  let description = [{
    LSTM Cell
    X    The input tensor with shape: [batch_size,
            
    H_t  The hidden state tensor at current time step with shape: [batch_size, hidden_size].
    
    C_t  The cell state tensor at current time step with shape: [batch_size, hidden_size].
    
    W    The weight tensor with shape: [4*hidden_size, input_size].
    
    R    The recurrence weight tensor with shape: [4*hidden_size, hidden_size].
        
    B    [Optional] The bias tensor for gates with shape: [4*hidden_size].
    
    P    [Optional] The weight tensor for peepholes with shape: 
                    [3*hidden_size] - 3 equals to only iof gates.
                    The order is: input, output, forget gates.
    
    hiddenSize    The number of hidden units for recurrent cell.
    
    weightsFormat The order of gates in weights tensors. 
                  The default format is IFCO since it is used by DNNL.
    
    activations   The vector of activation functions used inside
                  recurrent cell.
    
    activationsAlpha  The vector of alpha parameters for activation
                      functions in order respective to activation list.
    
    activationsBeta   The vector of beta parameters for activation
                      functions in order respective to activation list.
    
    clip              The value defining clipping range [-clip, clip] on
                      input of activation functions.
    
    inputForget       Controls coupling input and forget gates.
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ 
    return mlir::success(); /* TBD */ 
  }];  

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
        tblgen_state.addAttribute("activationAlpha", activationAlpha);
        tblgen_state.addAttribute("activationBeta", activationBeta);
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
    // get bias operand if present
    Value* B() 
    { 
      auto varArgs = optionalArgs();
      return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
    }
    // get peephole weights operand if present
    Value* P() 
    { 
      auto varArgs = optionalArgs();
      auto it = varArgs.begin();
      it = std::next(it);
      if (it == varArgs.end())
        return nullptr;
      it = std::next(it);
      return it == varArgs.end() ? *it : nullptr;
    }

    void setHiddenSize  (const Attribute& attr)       { this->setAttr("hiddenSize", attr); }
    void setActivations (const ArrayAttr& attr)       { this->setAttr("activation", attr); }
    void setActivationsAlpha (const ArrayAttr& attr)  { this->setAttr("activatiAlpha", attr); }
    void setActivationsBeta (const ArrayAttr& attr)   { this->setAttr("activatiBeta", attr); }
    void setClip(const Attribute& attr)               { this->setAttr("clip", attr); }
  }];
}

// LSTM Sequence Op
def NGLSTMSequenceOp : 
    NG_OneResult_Op<"lstmSeq", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$H_t, NG_TensorType:$C_t, 
               NG_TensorType:$S_l, NG_TensorType:$W, NG_TensorType:$R,
               NG_TensorType:$B, Variadic<NG_TensorType>:$optionalArgs, 
               I64Attr:$hiddenSize, 
               LSTMSeqDirectionsEnumAttr:$direction, 
               DefaultValuedAttr<LSTMWeightsFormatEnumAttr, 
                                 "static_cast<int32_t>(MLIRLSTMWeightsFormat::IFCO)">:$weightFormat,
               DefaultValuedAttr<StrArrayAttr, "{\"sigmoid\",\"tanh\",\"tanh\"}">:$activations,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationAlpha,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationBeta,
               DefaultValuedAttr<F32Attr, "0.0">:$clip,
               DefaultValuedAttr<BoolAttr, "false">:$inputForget)>
{
  let summary = "LSTM Sequence";
  let description = [{
        
    Class for lstm sequence node.
        
    It follows notation and equations defined as in ONNX standard:
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
        
    See: LSTMCell, RNNCell, GRUCell
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ 
    return mlir::success(); /* TBD */ 
  }];  

  let extraClassDeclaration = [{
    void setHiddenSize  (const Attribute& attr)       { this->setAttr("hiddenSize", attr); }
    void setActivations (const ArrayAttr& attr)       { this->setAttr("activation", attr); }
    void setActivationsAlpha (const ArrayAttr& attr)  { this->setAttr("activatiAlpha", attr); }
    void setActivationsBeta (const ArrayAttr& attr)   { this->setAttr("activatiBeta", attr); }
    void setClip(const Attribute& attr)               { this->setAttr("clip", attr); }

    Value* P() 
    { 
      auto varArgs = optionalArgs();
      return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
    }
  }];
}

// GRU Cell Op
def NGGRUCellOp : 
    NG_OneResult_Op<"gruCell", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$X, NG_TensorType:$W, NG_TensorType:$R,
               NG_TensorType:$H_t, Variadic<NG_TensorType>:$optionalArgs,
               I64Attr:$hiddenSize,
               DefaultValuedAttr<StrArrayAttr, "{\"sigmoid\",\"tanh\"}">:$activations,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationAlpha,
               DefaultValuedAttr<F32ArrayAttr, "{}">:$activationBeta,
               DefaultValuedAttr<F32Attr, "0.0">:$clip,
               DefaultValuedAttr<BoolAttr, "false">:$linearBeforeReset)>
{
  let summary = "This class represents only single *cell* and not whole GRU *layer*";
  let description = [{
    X                 The input tensor with shape: [batch_size, input_size].
    W                 The weight tensor with shape:
                      [gates_count * hidden_size, input_size].
    R                 The recurrence weight tensor with shape:
                      [gates_count * hidden_size, hidden_size].
    H_t               The hidden state tensor at current time step with
                      shape: [batch_size, hidden_size].
    hidden_size       The number of hidden units for recurrent cell.
    B[Optional]       The bias tensor for input gate with shape:
                      [2 * gates_count * hidden_size].
    activations       The vector of activation functions used inside
                      recurrent cell.
    activation_alpha  The vector of alpha parameters for activation
                      functions in order respective to activation list.
    activation_beta   The vector of beta parameters for activation functions
                      in order respective to activation list.
    clip              The value defining clipping range [-clip, clip] on
                      input of activation functions.
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
  let verifier = [{ return mlir::success(); /* TBD */ }];  

  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *X, Value* W, Value* R, Value* H_t,"
      "Attribute hiddenSize, ArrayAttr activations,"
      "ArrayAttr activationAlpha, ArrayAttr activationBeta,"
      "Attribute clip, Attribute linearBeforeReset", [{
        tblgen_state.addOperands({X, W, R, H_t});
        tblgen_state.addAttribute("hiddenSize", hiddenSize);
        tblgen_state.addAttribute("activations", activations);
        tblgen_state.addAttribute("activationAlpha", activationAlpha);
        tblgen_state.addAttribute("activationBeta", activationBeta);
        tblgen_state.addAttribute("linearBeforeReset", linearBeforeReset);
        tblgen_state.addTypes(res);
      }]>,

    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res,"
      "Value *X, Value* W, Value* R, Value* H_t,"
      "Attribute hiddenSize",
      [{
        tblgen_state.addOperands({X, W, R, H_t});
        tblgen_state.addAttribute("hiddenSize", hiddenSize);
        tblgen_state.addTypes(res);
      }]>
    ];

    let extraClassDeclaration = [{
      void setHiddenSize  (const Attribute& attr)       { this->setAttr("hiddenSize", attr); }
      void setActivations (const ArrayAttr& attr)       { this->setAttr("activation", attr); }
      void setActivationsAlpha (const ArrayAttr& attr)  { this->setAttr("activatiAlpha", attr); }
      void setActivationsBeta (const ArrayAttr& attr)   { this->setAttr("activatiBeta", attr); }
      void setLinearBeforeReset(const Attribute& attr)  { this->setAttr("linearBeforeReset", attr); }

      // get Bias operand if present
      Value* P() 
      { 
        auto varArgs = optionalArgs();
        return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
      }
    }];
}

// LayerNorm Op
// Op produces 3 results when keepStats is true
// Op is defined with 3 results, 2 of which are invalid/dead if keepStats is false
def NGLayerNormOp : 
    NG_Op<"layernorm", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Results<(outs NG_TensorType:$res, NG_TensorType:$mean, NG_TensorType:$var)>,
    Arguments<(ins NG_TensorType:$data, Variadic<NG_TensorType>:$optionalArgs,
               DefaultValuedAttr<BoolAttr, "true">:$keepStats,
               DefaultValuedAttr<I64Attr, "1">    :$beginNormAxis,
               DefaultValuedAttr<F32Attr, "1e-5"> :$epsilon)>
{
  let summary = "LayerNorm Op";
  let description = "Constructs a LayerNorm operation.";
    
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &tblgen_state, ArrayRef<Type> res,"
    "Value *data, Attribute keepStats, Attribute beginNormAxis, Attribute epsilon", [{
       tblgen_state.addOperands(data);
       tblgen_state.addAttribute("keepStats", keepStats);
       tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
       tblgen_state.addAttribute("epsilon", epsilon);
       tblgen_state.addTypes(res);
    }]>
  ];

  let extraClassDeclaration = [{
    // get Scale operand if present
    Value* Scale() 
    { 
      auto varArgs = optionalArgs();
      return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
    }
    // get Bias operand if present
    Value* Bias() 
    { 
      auto varArgs = optionalArgs();
      auto it = varArgs.begin();
      it = std::next(it);
      if (it == varArgs.end())
        return nullptr;
      it = std::next(it);
      return it == varArgs.end() ? *it : nullptr;
    }

    void setKeepstats(const Attribute& attr)      { this->setAttr("keepStats", attr);    }
    void setBeginNormAxis(const Attribute& attr)  { this->setAttr("beginNormAxis", attr);}
    void setEpsilonAxis(const Attribute& attr)    { this->setAttr("epsilon", attr);      }
  }];
}


// LayerNormBackprop Op
// Scale can be optional, in which case use a constant op of 0
def NGLayerNormBackpropOp : 
    NG_Op<"layernormBackprop", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Results<(outs NG_TensorType:$d_data, NG_TensorType:$d_scale, NG_TensorType:$d_bias)>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$delta, NG_TensorType:$scale, 
               Variadic<NG_TensorType>:$optionalArgs, 
               DefaultValuedAttr<I64Attr, "1">:$beginNormAxis,
               DefaultValuedAttr<F32Attr, "1e-5">:$epsilon)>
{
  let summary = "LayerNorm Op";
  let description = "Constructs an LayerNorm operation.";
    
  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, ArrayRef<Type> res,"
      "Value *data, Value *delta, Value *mean, Value *variance,"
      "Attribute beginNormAxis, Attribute epsilon", [{
        tblgen_state.addOperands({data, delta, mean, variance});
        tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
        tblgen_state.addAttribute("epsilon", epsilon);
        tblgen_state.addTypes(res);
      }]>,

    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, ArrayRef<Type> res,"
      "Value *data, Value *delta, Value *scale,"
      "Attribute beginNormAxis, Attribute epsilon", [{
        tblgen_state.addOperands({data, delta, scale});
        tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
        tblgen_state.addAttribute("epsilon", epsilon);
        tblgen_state.addTypes(res);
      }]>,

    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, ArrayRef<Type> res,"
      "Value *data, Value *delta,"
      "Attribute beginNormAxis, Attribute epsilon", [{
        tblgen_state.addOperands({data, delta});
        tblgen_state.addAttribute("beginNormAxis", beginNormAxis);
        tblgen_state.addAttribute("epsilon", epsilon);
        tblgen_state.addTypes(res);
      }]>,
  ];

  let extraClassDeclaration = [{
    // get Mean operand if present
    Value* Mean() 
    { 
      auto varArgs = optionalArgs();
      return varArgs.begin() != varArgs.end() ? *varArgs.begin() : nullptr;
    }
    // get Variance operand if present
    Value* Variance() 
    { 
      auto varArgs = optionalArgs();
      auto it = varArgs.begin();
      it = std::next(it);
      if (it == varArgs.end())
        return nullptr;
      it = std::next(it);
      return it == varArgs.end() ? *it : nullptr;
    }

    void setBeginNormAxis(const Attribute& attr)  { this->setAttr("beginNormAxis", attr);}
    void setEpsilonAxis(const Attribute& attr)    { this->setAttr("epsilon", attr);      }
  }];
}

// HardSigmoid Op
def NGHardSigmoid : 
    NG_OneResult_Op<"hardSigmoid", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data,
               F32Attr:$alpha, F32Attr:$beta)>
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

// Gemm Op
def NGGemmOp : 
    NG_OneResult_Op<"gemm", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$A, NG_TensorType:$B, NG_TensorType:$C,
              DefaultValuedAttr<F32Attr, "1.0">:$alpha,
              DefaultValuedAttr<F32Attr, "1.0">:$beta,
              DefaultValuedAttr<BoolAttr, "false">:$transA,
              DefaultValuedAttr<BoolAttr, "false">:$transB)>
{
    let summary = "Gemm Op";
    let description = [{
      A' = transpose(A) if transA else A
      B' = transpose(B) if transB else B
      Compute Y = alpha * A' * B' + beta * C
    }];
    
    let extraClassDeclaration = [{
      void setAlpha(const Attribute& attr)   { this->setAttr("alpha", attr);}
      void setBeta(const Attribute& attr)    { this->setAttr("beta", attr); }

      void setTransA(const Attribute& attr)  { this->setAttr("transA", attr); }
      void setTransB(const Attribute& attr)  { this->setAttr("transB", attr); }
    }];
}
                  
// GroupConv Op
def NGGroupConvOp : 
    NG_OneResult_Op<"groupConv", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$images, NG_TensorType:$filters, 
               I64ArrayAttr:$strides,
               I64ArrayAttr:$padBelow,
               I64ArrayAttr:$padAbove,
               I64Attr:$groups,
               DefaultValuedAttr<PadTypeEnumAttr, 
                                "static_cast<int64_t>(MLIRPadType::EXPLICIT)">:$padType)>
{
  let summary = "Group Convolution Op";
  let description = [{
    Group Convolution
  }];

  let builders = [
    // Builder without padType
    OpBuilder<
      "Builder *builder, OperationState &tblgen_state, Type res, Value *images,"
      "Value *filters, ArrayAttr strides, ArrayAttr padBelow, ArrayAttr padAbove,"
      "Attribute groups",
      [{
        tblgen_state.addOperands({images, filters});
        tblgen_state.addAttribute("strides", strides);
        tblgen_state.addAttribute("padBelow", padBelow);
        tblgen_state.addAttribute("padAbove", padAbove);
        tblgen_state.addAttribute("groups", groups);
        tblgen_state.addTypes(res);
      }]>
  ];

  let extraClassDeclaration = [{
    void setStrides(const ArrayAttr& attr)  { this->setAttr("strides", attr);  }
    void setPadAbove(const ArrayAttr& attr) { this->setAttr("padAbove", attr); }
    void setPadBelow(const ArrayAttr& attr) { this->setAttr("padBelow", attr); }
    void setPadType(const Attribute& attr)  { this->setAttr("padType", attr);  }
  }];
}

// GroupConvTranspose Op
def NGGroupConvTransposeOp :
  NG_OneResult_Op<"groupConvTranspose", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
  Arguments<(ins NG_TensorType:$images, NG_TensorType:$filters,
             I64ArrayAttr:$strides, I64ArrayAttr:$padBelow, I64ArrayAttr:$padAbove, 
             I64ArrayAttr:$outputPad, 
             DefaultValuedAttr<I64Attr, "1UL">:$groups, 
             DefaultValuedAttr<PadTypeEnumAttr, 
                               "static_cast<int64_t>(MLIRPadType::EXPLICIT)">:$padType,
             I64ArrayAttr:$outputShape)>
{
  let summary = "Group Transpose Convolution (Deconvolution)";
  let description = [{
    images          The node producing input images data.
    filters         The node producing filters data.
    strides         The strides along each feature axis.
    padBelow        The padding added at the beggining of each feature axis.
    padAbove        The padding added at the end of each feature axis.
    outputPad       The zero-padding (adjustment) added to one side of the
                    output.
    groups          The number of groups the input channels and output
                    channels are divided into.
    padType        The provided padding type.
    outputShape    The output shape. When provided padding values are
                    automatically inferred.
  }];

  let builders = [
    OpBuilder<"Builder *builder, OperationState &tblgen_state, Type res,"
              "Value *images, Value *filters, Attribute groups", [{
      tblgen_state.addOperands({images, filters});
      tblgen_state.addAttribute("groups", groups);
      tblgen_state.addTypes(res);
    }]>,
    OpBuilder<"Builder *builder, OperationState &tblgen_state, Type res,"
              "Value *images, Value *filters", [{
      tblgen_state.addOperands({images, filters});
      tblgen_state.addTypes(res);
    }]>,
    OpBuilder<"Builder *builder, OperationState &tblgen_state, Type res,"
              "Value *images, Value *filters, ArrayAttr strides,"
              "ArrayAttr outputPad, ArrayAttr outputShape,"
              "Attribute groups", [{
      tblgen_state.addOperands({images, filters});
      tblgen_state.addAttribute("strides", strides);
      tblgen_state.addAttribute("outputPad", outputPad);
      tblgen_state.addAttribute("outputShape", outputShape);
      tblgen_state.addAttribute("groups", groups);
    }]>,
    OpBuilder<"Builder *builder, OperationState &tblgen_state, Type res,"
              "Value *images, Value *filters,"
              "ArrayAttr outputShape, Attribute groups", [{
      tblgen_state.addOperands({images, filters});
      tblgen_state.addAttribute("outputShape", outputShape);
      tblgen_state.addAttribute("groups", groups);
    }]>
  ];

  let extraClassDeclaration = [{
    void setStrides(const ArrayAttr& attr)    { this->setAttr("strides", attr);  }
    void setPadAbove(const ArrayAttr& attr)   { this->setAttr("padAbove", attr); }
    void setPadBelow(const ArrayAttr& attr)   { this->setAttr("padBelow", attr); }
    void setPadType(const Attribute& attr)    { this->setAttr("padType", attr);  }
    void setOutputPad(const ArrayAttr& attr)  { this->setAttr("outputPad", attr);}
    void setOutputShape(const ArrayAttr& attr){ this->setAttr("outputShape", attr);}
  }];
}
                      

// GRN Op
def NGGRNOp :
    NG_OneResult_Op<"grn", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, 
               DefaultValuedAttr<F32Attr, "1.0">:$bias)>
{
  let summary = "GRN Op";
  let description = [{
    Global Response Normalization with L2 norm (across channels only)
    data  - Node producing the input tensor
    bias  - The bias added to the variance.
  }];

  let extraClassDeclaration = [{
    void setBias(const Attribute& attr) { this->setAttr("bias", attr); }
  }];
}

// Clamp Op
def NGClampOp :
    NG_OneResult_Op<"clamp", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, F64Attr:$min, F64Attr:$max)>
{
  let summary = "Clamp Op";
  let description = [{
    Performs a clipping operation on all elements of the input node
    
    All input values that are outside of the <min;max> range are set to 'min' or 'max'
    depending on which side of the <min;max> range they are. The values that fall into
    this range remain unchanged.
  }];

  let extraClassDeclaration = [{
    void setMin(const Attribute& attr) { this->setAttr("min", attr); }
    void setMax(const Attribute& attr) { this->setAttr("max", attr); }
  }];
}

// Gelu Op
def NGGeluOp :
    NG_OneResult_Op<"gelu", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data)>
{
  let summary = "Gelu Op";
  let description = [{
    Gaussian Error Linear Unit
    f(x) = 0.5 * x * (1 + erf( x / sqrt(2) )
  }];
}

// GeluBackpropFactor Op
def NGGeluBackpropFactorOp :
    NG_OneResult_Op<"geluBackpropFactor", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data)>
{
  let summary = "Gelu Backprop Op";
  let description = [{
    Backprop for Gelu(x) is GeluBackprop(x) * delta
  }];
}

// Elu Op
def NGEluOp :
    NG_OneResult_Op<"elu", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, F64Attr:$alpha)>
{
  let summary = "Elu Op";
  let description = [{
    Exponential Linear Unit
    x <  0 => f(x) = alpha * (exp(x) - 1.)
    x >= 0 => f(x) = x
  }];

  let extraClassDeclaration = [{
    void setAlpha(const Attribute& attr) { this->setAttr("alpha", attr); }
  }];
}

// FakeQuant Op
def NGFakeQuantOp :
    NG_OneResult_Op<"fakeQuant", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$inputLow, NG_TensorType:$inputHigh, 
               NG_TensorType:$outputLow, NG_TensorType:$outputHigh, 
               I64Attr:$levels,
               DefaultValuedAttr<AutoBroadcastEnumAttr,
                                "static_cast<int64_t>(MLIRPadType::EXPLICIT)">:$autoBroadcast)>
{
  let summary = "Op performing element-wise linear quantization.";
  let description = [{
    Input floating point values are quantized into a discrete
    set of floating point values.
    
    Implementation This class creates a node which performs the following
    operation:
      round((data - input_low) / (input_high - input_low) * (levels-1)) /
      (levels-1) * (output_high - output_low) + output_low
  }];

  let extraClassDeclaration = [{
    void setLevels(const Attribute& attr) { this->setAttr("levels", attr); }
    void setAutoBroadcast(const Attribute& attr) { this->setAttr("autoBroadcast", attr); }
  }];
}

// DepthToSpace Op
def NGDepthToSpaceOp : 
    NG_OneResult_Op<"depthToSpace", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$data, 
               DefaultValuedAttr<I64Attr, "1">:$blockSize, 
               DepthSpaceModeEnumAttr:$mode)>
{
  let summary = "DepthToSpace Op";
  let description = [{
    DepthToSpace permutes data from the depth dimension of the input blob into
    spatial dimensions.
    Values from the depth dimension (assuming NCHW layout) are moved in
    spatial blocks to the height and width dimensions.
    Output node produces a tensor with shape:
    [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
  }];

  let extraClassDeclaration = [{
    void setBlockSize(const Attribute& attr) { this->setAttr("blockSize", attr);}
    void setMode(const Attribute& attr)      { this->setAttr("mode", attr);     } 
  }];
}

// ConvolutionBias Op
def NGConvBiasOp :
    NG_OneResult_Op<"convBias", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$images, NG_TensorType:$filters, NG_TensorType:$bias,
               I64ArrayAttr:$strides, I64ArrayAttr:$padBelow, I64ArrayAttr:$padAbove,
               DefaultValuedAttr<BoolAttr, "false">:$withRelu)>
{
  let summary = "Convolution Bias Op";
  let description = "Convolution + bias forward prop for batched convolution operation.";

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &tblgen_state, Type res,"
    "Value *images, Value *filters, Value *bias, Attribute withRelu", [{
       tblgen_state.addOperands({images, filters, bias});
       tblgen_state.addAttribute("withRelu", withRelu);
       tblgen_state.addTypes(res);
     }]>,

    OpBuilder<
    "Builder *builder, OperationState &tblgen_state, Type res,"
    "Value *images, Value *filters, Value *bias", [{
       tblgen_state.addOperands({images, filters, bias});
       tblgen_state.addTypes(res);
     }]>
    ];

  let extraClassDeclaration = [{
    void setStrides(const ArrayAttr& attr)  { this->setAttr("strides", attr);  }
    void setPadAbove(const ArrayAttr& attr) { this->setAttr("padAbove", attr); }
    void setPadBelow(const ArrayAttr& attr) { this->setAttr("padBelow", attr); }
    void setWithRelu(const Attribute& attr)  {this->setAttr("withRelu", attr); }
  }];
}

// ConvBiasBackpropFiltersBias Op
def NGConvBiasBackpropFiltersBias :
    NG_Op<"convBiasBackpropFiltersBias", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Results<(outs NG_TensorType:$filter, NG_TensorType:$bias)>, 
    Arguments<(ins NG_TensorType:$images, NG_TensorType:$outputDelta,
               I64ArrayAttr:$filtersShape, I64ArrayAttr:$biasShape,
               I64ArrayAttr:$strides, I64ArrayAttr:$padBelow, I64ArrayAttr:$padAbove)>
{
  let extraClassDeclaration = [{
    void setFiltersShape(const ArrayAttr& attr) { this->setAttr("filtersShape", attr); }
    void setBiasShape(const ArrayAttr& attr)    { this->setAttr("biasShape", attr);    }

    void setStrides(const ArrayAttr& attr)  { this->setAttr("strides", attr);  }
    void setPadAbove(const ArrayAttr& attr) { this->setAttr("padAbove", attr); }
    void setPadBelow(const ArrayAttr& attr) { this->setAttr("padBelow", attr); }
  }];
}

// ConvBiasAdd Op
def NGConvBiasAddOp :
    NG_OneResult_Op<"convBiasAdd", [NoSideEffect, DeclareOpInterfaceMethods<FusedOp>]>,
    Arguments<(ins NG_TensorType:$images, NG_TensorType:$filters, 
               NG_TensorType:$bias, NG_TensorType:$sumInput,
               I64ArrayAttr:$strides, I64ArrayAttr:$padBelow, I64ArrayAttr:$padAbove,
               DefaultValuedAttr<BoolAttr, "false">:$withRelu)>
{
  let summary = "Convolution Bias Add Op";
  let description = "Convolution + bias + add forward prop for batched convolution operation.";

  let extraClassDeclaration = [{
    void setStrides(const ArrayAttr& attr)  { this->setAttr("strides", attr);  }
    void setPadAbove(const ArrayAttr& attr) { this->setAttr("padAbove", attr); }
    void setPadBelow(const ArrayAttr& attr) { this->setAttr("padBelow", attr); }
    void setWithRelu(const Attribute& attr)  {this->setAttr("withRelu", attr); }
  }];
}


#endif //NG_FUSED_OPS
