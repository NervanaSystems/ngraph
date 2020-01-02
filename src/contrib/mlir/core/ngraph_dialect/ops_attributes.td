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

def PadModeEnumAttr : I32EnumAttr<"MLIRPadMode", 
                                  "Padding modes for pad operator", 
                                  [PadModeConstant, PadModeEdge, PadModeReflect, PadModeSymmetric]>;

// Sort Types for TopK
def SortTypeNone       : I32EnumAttrCase<"NONE",    0>;
def SortTypeIndices    : I32EnumAttrCase<"INDICES", 1>;
def SortTypeValues     : I32EnumAttrCase<"VALUES",  2>;

def SortTypeEnumAttr   : I32EnumAttr<"MLIRSortType", "Sort types for topk operator", 
                                     [SortTypeNone, SortTypeIndices, SortTypeValues]>;

// Modes for normalizeL2
def EpsModeAdd          : I32EnumAttrCase<"ADD", 0>;
def EpsModeMax          : I32EnumAttrCase<"MAX", 1>;

def EpsModeEnumAttr     : I32EnumAttr<"MLIREpsMode", 
                                      "Specifies how eps is combined with L2 value",
                                      [EpsModeAdd, EpsModeMax]>;


def AutoBroadcastNone       : I32EnumAttrCase<"NONE",       0>;
def AutoBroadcastExplicit   : I32EnumAttrCase<"EXPLICIT",   1>;
def AutoBroadcastNumPy      : I32EnumAttrCase<"NUMPY",      2>;
def AutoBroadcastPDPD       : I32EnumAttrCase<"PDPD",       3>;

def AutoBroadcastEnumAttr   : I32EnumAttr<"MLIRAutoBroadcastMode",
                                          "Specifies auto-broadcast for an op",
                                          [AutoBroadcastNone, AutoBroadcastExplicit,
                                           AutoBroadcastNumPy, AutoBroadcastPDPD]>;


def DepthSpaceModeBlocks  : I32EnumAttrCase<"BLOCKS_FIRST", 0>;
def DepthSpaceModeDepth   : I32EnumAttrCase<"DEPTH_FIRST",  1>;

def DepthSpaceModeEnumAttr: I32EnumAttr<"MLIRDepthToSpaceMode", 
                                        "Specifies how the input depth dimension is split to block coordinates",
                                        [DepthSpaceModeBlocks, DepthSpaceModeDepth]>;

def LSTMWeightsFormatFICO   : I32EnumAttrCase<"FICO", 0>;   // IE
def LSTMWeightsFormatICOF   : I32EnumAttrCase<"ICOF", 1>;   // PyTorch
def LSTMWeightsFormatIFCO   : I32EnumAttrCase<"IFCO", 2>;   // DNNL, TF, MxNet
def LSTMWeightsFormatIFOC   : I32EnumAttrCase<"IFOC", 3>;   // Caffe
def LSTMWeightsFormatIOFC   : I32EnumAttrCase<"IOFC", 4>;   // ONNX

def LSTMWeightsFormatEnumAttr: I32EnumAttr<"MLIRLSTMWeightsFormat", 
                                            "LSTM Cell Weights Format",
                                            [LSTMWeightsFormatFICO, LSTMWeightsFormatICOF, 
                                             LSTMWeightsFormatIFCO, LSTMWeightsFormatIFOC,
                                             LSTMWeightsFormatIOFC]>;

def LSTMSeqDirectionFWD     : I32EnumAttrCase<"FORWARD", 0>;
def LSTMSeqDirectionRVS     : I32EnumAttrCase<"REVERSE", 1>;
def LSTMSeqDirectionBID     : I32EnumAttrCase<"BIDIRECTIONAL", 2>;
            
def LSTMSeqDirectionsEnumAttr: I32EnumAttr<"MLIRLSTMSeqDirection", 
                                           "LSTM Sequence Direction", 
                                            [LSTMSeqDirectionFWD, LSTMSeqDirectionRVS, 
                                             LSTMSeqDirectionBID]>;
#endif // NG_OP_ATTRIBUTES
