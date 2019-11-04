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
// This is the nGraph Dialect operation definition file.
//
//===----------------------------------------------------------------------===//

#ifdef NG_OP_ATTRIBUTES
#else
#define NG_OP_ATTRIBUTES

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// Padding Type used for `Convolution` and `Pooling`
//
// Follows ONNX padding type definitions
// EXPLICIT   - Pad dimensions are explicity specified
// SAME_LOWER - Pad dimensions computed to match input shape
//              Ceil(num_dims/2) at the beginning and
//              Floor(num_dims/2) at the end
// SAME_UPPER - Pad dimensions computed to match input shape
//              Floor(num_dims/2) at the beginning and
//              Ceil(num_dims/2) at the end
// VALID      - No padding

def PadTypeExplicit     : I32EnumAttrCase<"EXPLICIT",   0>;
def PadTypeNotSet       : I32EnumAttrCase<"NOT_SET",    1>;
def PadTypeSameLower    : I32EnumAttrCase<"SAME_LOWER", 2>;
def PadTypeSameUpper    : I32EnumAttrCase<"SAME_UPPER", 3>;
def PadTypeAuto         : I32EnumAttrCase<"AUTO",       4>;
def PadTypeValid        : I32EnumAttrCase<"VALID",      5>;

def PadTypeEnumAttr : I32EnumAttr<"MLIRPadType", "Padding Type used for Convolution and pooling", 
                                  [PadTypeExplicit, PadTypeNotSet, PadTypeSameLower, 
                                   PadTypeSameUpper, PadTypeAuto, PadTypeValid]>;

// Modes for the `Pad` operator
def PadModeConstant : I32EnumAttrCase<"CONSTANT",  0> ;
def PadModeEdge     : I32EnumAttrCase<"EDGE",      1> ;
def PadModeReflect  : I32EnumAttrCase<"REFLECT",   2> ;
def PadModeSymmetric: I32EnumAttrCase<"SYMMETRIC", 3> ;

def PadModeEnumAttr : I32EnumAttr<"MLIRPadMode", "Padding modes for pad operator", 
                                  [PadModeConstant, PadModeEdge, PadModeReflect, PadModeSymmetric]>;

// Sort Types for TopK
def SortTypeNone       : I32EnumAttrCase<"NONE",    0>;
def SortTypeIndices    : I32EnumAttrCase<"INDICES", 1>;
def SortTypeValues     : I32EnumAttrCase<"VALUES",  2>;

def SortTypeEnumAttr   : I32EnumAttr<"MLIRSortType", "Sort types for topk operator", 
                                     [SortTypeNone, SortTypeIndices, SortTypeValues]>;

#endif // NG_OP_ATTRIBUTES
