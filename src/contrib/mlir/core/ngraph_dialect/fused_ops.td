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
    NG_OneResult_Op<"squeeze", [NoSideEffect, FusedOp]>,
    Arguments<(ins NG_TensorType:$data, NG_TensorType:$axes)>
{
  let summary = "Squeeze Op";
  let description = [{
    Squeeze Op
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

  let verifier = [{ return verifyOp(this); }];

  let extraClassDeclaration = [{
    void decompose()  { 
        //TODO: Call a templatized helper: decompose(this) to do the actual decomposition
    }
    
  }];
}
#endif //NG_FUSED_OPS
