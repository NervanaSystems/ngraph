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

#ifndef NGRAPH_OPS
#define NGRAPH_OPS

include "NgraphDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ngraph Operations
//===----------------------------------------------------------------------===//

def Ngraph_AddOp : Ngraph_Op<"Add", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Performs elementwise addition on the inputs.";

    let description = [{
        Add tensors
    }];

    let arguments = (ins
        AnyType:$input1,
        AnyType:$input2
    );

    let results = (outs
        AnyType:$res
    );

    // Ngraph_DerivedOperandTypeAttr T = Ngraph_DerivedOperandTypeAttr<0>;
}

#endif // NGRAPH_OPS
