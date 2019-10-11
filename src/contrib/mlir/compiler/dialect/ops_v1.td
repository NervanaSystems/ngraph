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
// This is the nGraph Dialect Version 1 operations definition file.
// All Operations in this file implement OpVersion1 interface.
//===----------------------------------------------------------------------===//

// Gather
def NGGatherV1Op : 
    NG_OneResult_Op<"gather.v1", [NoSideEffect, OpVersion1]>,
    Arguments<(ins NG_TensorType:$params, NG_TensorType:$indices, NG_TensorType:$axis)>
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
