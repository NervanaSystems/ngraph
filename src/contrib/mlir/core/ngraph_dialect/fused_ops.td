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
def NGSpaceToDepth :
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
def NGShuffleChannels :
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
def NGScaleShift :
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
def NGRNNCell : 
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
#endif //NG_FUSED_OPS
