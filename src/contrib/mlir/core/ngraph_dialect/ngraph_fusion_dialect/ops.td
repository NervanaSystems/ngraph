//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

def NG_Fusion_Dialect : Dialect {
	let name = "ngfusion";
	let cppNamespace = ""; 
}

// Tensor type of the fusion, we will check if the type is same as nGrph tensor
// type
def NG_Fusion_TensorType : Type<CPred<"$_self.isa<mlir::NGTensorType>()">,
                     "nGraph Tensor Type">;

// Base class for the NG_Fusion dialect Op
class NG_Fusion_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<NG_Fusion_Dialect, mnemonic, traits> {}

// Operations producing single result.
// Will set OneResult trait based on Results out dag.
class NG_Fusion_OneResult_Op<string mnemonic, list<OpTrait> traits = []> :
      NG_Fusion_Op<mnemonic, traits>, Results<(outs NG_Fusion_TensorType:$res)> {}

// class for fusing Binary ElementWise Ops
def NG_FusionBinaryElementWiseOp :
	NG_Fusion_OneResult_Op<"FusedBinaryElementWise", [NoSideEffect]>,
    Arguments<(ins NG_Fusion_TensorType:$arg1, NG_Fusion_TensorType:$arg2)>
{
  let summary = "Fused Binary ElementWise Op";
  let description = [{
    Fusion of Bianry ElementWise Op's sharing common inputs
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
}

// class for fusing Unary ElementWise Ops
def NG_FusionUnaryElementWiseOp :
	NG_Fusion_OneResult_Op<"FusedUnaryElementWise", [NoSideEffect]>,
    Arguments<(ins NG_Fusion_TensorType:$arg)>
{
  let summary = "Fused Unary ElementWise Op";
  let description = [{
    Fusion of Unary ElementWise Op's sharing common inputs
  }];

  let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];
}
