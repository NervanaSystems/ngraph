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
// This is the nGraph Dialect operation interfaces definitions
//
//===----------------------------------------------------------------------===//

#ifdef NG_OP_INTERFACES
#else
#define NG_OP_INTERFACES

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// Op Interfaces for Op Versions
// They are empty for now. To check the version of an op, we do:
// 	Operation *op = …;
//	if (dyn_cast<OpVersion0Interface>(op))

def OpVersion0 : OpInterface<"OpVersion0"> {
    let description=[{
        Interface for Version 0 Ops
    }];
    // Interface is empty for now. 
}

def OpVersion1 : OpInterface<"OpVersion1"> {
    let description=[{
        Interface for Version 1 Ops
    }];
    // Interface is empty for now. 
}

def FusedOp : OpInterface<"FusedOp"> {
    let description=[{
        Interface for fused ops. 
        Provides an API to decompose an op
    }];
   
    let methods = [
        InterfaceMethod<
            "Decompose the operation",
            "void",
            "decompose"
        >
    ];
}

#endif // NG_OP_INTERFACES
