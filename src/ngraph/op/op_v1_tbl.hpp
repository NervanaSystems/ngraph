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

// This collection contains one entry for each op. If an op is added it must be
// added to this list.
//
// In order to use this list you want to define a macro named exactly NGRAPH_OP
// When you are done you should undef the macro
// As an example if you wanted to make a list of all op names as strings you could do this:
//
// #define NGRAPH_OP(a,b) #a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// "Abs",
// "Acos",
// ...
//
// #define NGRAPH_OP(a,b) b::a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// ngraph::op::Abs,
// ngraph::op::Acos,
// ...
//
// It's that easy. You can use this for fun and profit.

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(Abs, ngraph::op)
NGRAPH_OP(Acos, ngraph::op)
// NGRAPH_OP(Acosh, ngraph::op)
NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(Asin, ngraph::op)
// NGRAPH_OP(Asinh, ngraph::op)
NGRAPH_OP(Atan, ngraph::op)
// NGRAPH_OP(Atanh, ngraph::op)
NGRAPH_OP(AvgPool, ngraph::op::v1)
NGRAPH_OP(BatchNormInference, ngraph::op)
NGRAPH_OP(BinaryConvolution, ngraph::op::v1)
NGRAPH_OP(Broadcast, ngraph::op::v1)
// NGRAPH_OP(CTCGreedyDecoder, ngraph::op)
NGRAPH_OP(Ceiling, ngraph::op)
NGRAPH_OP(Clamp, ngraph::op)
NGRAPH_OP(Concat, ngraph::op)
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Convert, ngraph::op)
// NGRAPH_OP(ConvertLike, ngraph::op)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(Cos, ngraph::op)
NGRAPH_OP(Cosh, ngraph::op)
// NGRAPH_OP(DeformableConvolution, ngraph::op)
// NGRAPH_OP(DeformablePSROIPooling, ngraph::op)
NGRAPH_OP(DepthToSpace, ngraph::op)
// NGRAPH_OP(DetectionOutput, ngraph::op)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Elu, ngraph::op)
NGRAPH_OP(Erf, ngraph::op)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(Exp, ngraph::op)
NGRAPH_OP(FakeQuantize, ngraph::op)
NGRAPH_OP(Floor, ngraph::op)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Gather, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEq, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op)
// NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op)
NGRAPH_OP(HardSigmoid, ngraph::op)
NGRAPH_OP(Interpolate, ngraph::op)
// NGRAPH_OP(LeakyRelu, ngraph::op)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(Log, ngraph::op)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(LRN, ngraph::op)
NGRAPH_OP(LSTMCell, ngraph::op)
NGRAPH_OP(LSTMSequence, ngraph::op)
NGRAPH_OP(MatMul, ngraph::op)
NGRAPH_OP(MaxPool, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Mod, ngraph::op)
NGRAPH_OP(Multiply, ngraph::op::v1)
NGRAPH_OP(Negative, ngraph::op)
// NGRAPH_OP(NonMaxSuppression, ngraph::op)
NGRAPH_OP(NormalizeL2, ngraph::op)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(OneHot, ngraph::op::v1)
NGRAPH_OP(PRelu, ngraph::op)
// NGRAPH_OP(PSROIPooling, ngraph::op)
NGRAPH_OP(Pad, ngraph::op::v1)
NGRAPH_OP(Parameter, ngraph::op)
NGRAPH_OP(Power, ngraph::op::v1)
// NGRAPH_OP(PriorBox, ngraph::op)
// NGRAPH_OP(PriorBoxClustered, ngraph::op)
// NGRAPH_OP(Proposal, ngraph::op)
NGRAPH_OP(Range, ngraph::op)
NGRAPH_OP(Relu, ngraph::op)
// NGRAPH_OP(ReduceLogicalAnd, ngraph::op)
// NGRAPH_OP(ReduceLogicalOr, ngraph::op)
NGRAPH_OP(ReduceMax, ngraph::op::v1)
// NGRAPH_OP(ReduceMean, ngraph::op)
NGRAPH_OP(ReduceMin, ngraph::op::v1)
NGRAPH_OP(ReduceProd, ngraph::op::v1)
NGRAPH_OP(ReduceSum, ngraph::op::v1)
// NGRAPH_OP(RegionYolo, ngraph::op)
NGRAPH_OP(Reshape, ngraph::op::v1)
NGRAPH_OP(Result, ngraph::op)
NGRAPH_OP(Reverse, ngraph::op::v1)
NGRAPH_OP(ReverseSequence, ngraph::op)
NGRAPH_OP(RNNCell, ngraph::op)
// NGRAPH_OP(ROIPooling, ngraph::op)
NGRAPH_OP(ShapeOf, ngraph::op)
NGRAPH_OP(ShuffleChannels, ngraph::op)
NGRAPH_OP(Sign, ngraph::op)
NGRAPH_OP(Sigmoid, ngraph::op)
NGRAPH_OP(Sin, ngraph::op)
NGRAPH_OP(Sinh, ngraph::op)
NGRAPH_OP(Softmax, ngraph::op::v1)
NGRAPH_OP(Sqrt, ngraph::op)
NGRAPH_OP(SpaceToDepth, ngraph::op)
NGRAPH_OP(Split, ngraph::op)
NGRAPH_OP(SquaredDifference, ngraph::op)
NGRAPH_OP(Squeeze, ngraph::op)
NGRAPH_OP(StridedSlice, ngraph::op::v1)
NGRAPH_OP(Subtract, ngraph::op)
NGRAPH_OP(Tan, ngraph::op)
NGRAPH_OP(Tanh, ngraph::op)
NGRAPH_OP(TensorIterator, ngraph::op)
NGRAPH_OP(Tile, ngraph::op)
NGRAPH_OP(TopK, ngraph::op::v1)
NGRAPH_OP(Transpose, ngraph::op)
NGRAPH_OP(Unsqueeze, ngraph::op)
NGRAPH_OP(VariadicSplit, ngraph::op::v1)

// Related to v1
NGRAPH_OP(AvgPoolBackprop, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropFilters, ngraph::op::v1)
NGRAPH_OP(MaxPoolBackprop, ngraph::op::v1)

// Other
NGRAPH_OP(GenerateMask, ngraph::op::v1)
