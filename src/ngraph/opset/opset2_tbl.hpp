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

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(Abs, ngraph::op::v0)
NGRAPH_OP(Acos, ngraph::op::v0)
NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(Asin, ngraph::op::v0)
NGRAPH_OP(Atan, ngraph::op::v0)
NGRAPH_OP(AvgPool, ngraph::op::v1)
NGRAPH_OP(BatchNormInference, ngraph::op::v0)
NGRAPH_OP(BinaryConvolution, ngraph::op::v1)
NGRAPH_OP(Broadcast, ngraph::op::v1)
NGRAPH_OP(CTCGreedyDecoder, ngraph::op::v0)
NGRAPH_OP(Ceiling, ngraph::op::v0)
NGRAPH_OP(Clamp, ngraph::op::v0)
NGRAPH_OP(Concat, ngraph::op::v0)
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Convert, ngraph::op::v0)
NGRAPH_OP(ConvertLike, ngraph::op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(Cos, ngraph::op::v0)
NGRAPH_OP(Cosh, ngraph::op::v0)
NGRAPH_OP(DeformableConvolution, ngraph::op::v1)
NGRAPH_OP(DeformablePSROIPooling, ngraph::op::v1)
NGRAPH_OP(DepthToSpace, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, ngraph::op::v0)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Elu, ngraph::op::v0)
NGRAPH_OP(Erf, ngraph::op::v0)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(Exp, ngraph::op::v0)
NGRAPH_OP(FakeQuantize, ngraph::op::v0)
NGRAPH_OP(Floor, ngraph::op::v0)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Gather, ngraph::op::v1)
NGRAPH_OP(GatherTree, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEqual, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(GRN, ngraph::op::v0)
NGRAPH_OP(HardSigmoid, ngraph::op::v0)
NGRAPH_OP(Interpolate, ngraph::op::v0)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(Log, ngraph::op::v0)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(LRN, ngraph::op::v0)
NGRAPH_OP(LSTMCell, ngraph::op::v0)
NGRAPH_OP(LSTMSequence, ngraph::op::v0)
NGRAPH_OP(MatMul, ngraph::op::v0)
NGRAPH_OP(MaxPool, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Mod, ngraph::op::v1)
NGRAPH_OP(Multiply, ngraph::op::v1)

NGRAPH_OP(MVN, ngraph::op::v0) // Missing in opset1

NGRAPH_OP(Negative, ngraph::op::v0)
NGRAPH_OP(NonMaxSuppression, ngraph::op::v1)
NGRAPH_OP(NormalizeL2, ngraph::op::v0)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(OneHot, ngraph::op::v1)
NGRAPH_OP(PRelu, ngraph::op::v0)
NGRAPH_OP(PSROIPooling, ngraph::op::v0)
NGRAPH_OP(Pad, ngraph::op::v1)
NGRAPH_OP(Parameter, ngraph::op::v0)
NGRAPH_OP(Power, ngraph::op::v1)
NGRAPH_OP(PriorBox, ngraph::op::v0)
NGRAPH_OP(PriorBoxClustered, ngraph::op::v0)
NGRAPH_OP(Proposal, ngraph::op::v0)
NGRAPH_OP(Range, ngraph::op::v0)
NGRAPH_OP(Relu, ngraph::op::v0)
NGRAPH_OP(ReduceMax, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalAnd, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalOr, ngraph::op::v1)
NGRAPH_OP(ReduceMean, ngraph::op::v1)
NGRAPH_OP(ReduceMin, ngraph::op::v1)
NGRAPH_OP(ReduceProd, ngraph::op::v1)
NGRAPH_OP(ReduceSum, ngraph::op::v1)
NGRAPH_OP(RegionYolo, ngraph::op::v0)

NGRAPH_OP(ReorgYolo, ngraph::op::v0) // Missing in opset1

NGRAPH_OP(Reshape, ngraph::op::v1)
NGRAPH_OP(Result, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// NGRAPH_OP(Reverse, ngraph::op::v1)

NGRAPH_OP(ReverseSequence, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// NGRAPH_OP(RNNCell, ngraph::op::v0)

NGRAPH_OP(ROIPooling, ngraph::op::v0) // Missing in opset1

NGRAPH_OP(Select, ngraph::op::v1)
NGRAPH_OP(Selu, ngraph::op::v0)
NGRAPH_OP(ShapeOf, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// NGRAPH_OP(ShuffleChannels, ngraph::op::v0)

NGRAPH_OP(Sign, ngraph::op::v0)
NGRAPH_OP(Sigmoid, ngraph::op::v0)
NGRAPH_OP(Sin, ngraph::op::v0)
NGRAPH_OP(Sinh, ngraph::op::v0)
NGRAPH_OP(Softmax, ngraph::op::v1)
NGRAPH_OP(Sqrt, ngraph::op::v0)
NGRAPH_OP(SpaceToDepth, ngraph::op::v0)
NGRAPH_OP(Split, ngraph::op::v1)
NGRAPH_OP(SquaredDifference, ngraph::op::v0)
NGRAPH_OP(Squeeze, ngraph::op::v0)
NGRAPH_OP(StridedSlice, ngraph::op::v1)
NGRAPH_OP(Subtract, ngraph::op::v1)
NGRAPH_OP(Tan, ngraph::op::v0)
NGRAPH_OP(Tanh, ngraph::op::v0)
NGRAPH_OP(TensorIterator, ngraph::op::v0)
NGRAPH_OP(Tile, ngraph::op::v0)
NGRAPH_OP(TopK, ngraph::op::v1)
NGRAPH_OP(Transpose, ngraph::op::v1)
NGRAPH_OP(Unsqueeze, ngraph::op::v0)
NGRAPH_OP(VariadicSplit, ngraph::op::v1)

// Moved out of opset2, it was added to opset1 by mistake
// NGRAPH_OP(Xor, ngraph::op::v0)

// New operations added in opset2
NGRAPH_OP(Gelu, ngraph::op::v0)
NGRAPH_OP(BatchToSpace, ngraph::op::v1)
NGRAPH_OP(SpaceToBatch, ngraph::op::v1)
