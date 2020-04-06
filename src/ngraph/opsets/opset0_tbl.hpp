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
NGRAPH_OP(Add, ngraph::op)
NGRAPH_OP(All, ngraph::op)
NGRAPH_OP(AllReduce, ngraph::op)
NGRAPH_OP(And, ngraph::op)
NGRAPH_OP(Any, ngraph::op)
NGRAPH_OP(ArgMax, ngraph::op)
NGRAPH_OP(ArgMin, ngraph::op)
NGRAPH_OP(Asin, ngraph::op)
NGRAPH_OP(Atan, ngraph::op)
NGRAPH_OP(Atan2, ngraph::op)
NGRAPH_OP(AvgPool, ngraph::op)
NGRAPH_OP(AvgPoolBackprop, ngraph::op)
NGRAPH_OP(BatchMatMul, ngraph::op)
NGRAPH_OP(BatchMatMulTranspose, ngraph::op)
NGRAPH_OP(BatchNormInference, ngraph::op)
NGRAPH_OP(BatchNormTraining, ngraph::op)
NGRAPH_OP(BatchNormTrainingBackprop, ngraph::op)
NGRAPH_OP(Broadcast, ngraph::op)
NGRAPH_OP(BroadcastDistributed, ngraph::op)
NGRAPH_OP(BroadcastLike, ngraph::op)
NGRAPH_OP(Ceiling, ngraph::op)
NGRAPH_OP(Clamp, ngraph::op)
NGRAPH_OP(Concat, ngraph::op)
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Convert, ngraph::op)
NGRAPH_OP(Convolution, ngraph::op)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op)
NGRAPH_OP(ConvolutionBackpropFilters, ngraph::op)
NGRAPH_OP(ConvolutionBias, ngraph::op)
NGRAPH_OP(ConvolutionBiasAdd, ngraph::op)
NGRAPH_OP(ConvolutionBiasBackpropFiltersBias, ngraph::op)
NGRAPH_OP(Cos, ngraph::op)
NGRAPH_OP(Cosh, ngraph::op)
NGRAPH_OP(CrossEntropy, ngraph::op)
NGRAPH_OP(CrossEntropyBackprop, ngraph::op)
NGRAPH_OP(CropAndResize, ngraph::op)
NGRAPH_OP(CumSum, ngraph::op::v0)
NGRAPH_OP(DepthToSpace, ngraph::op)
NGRAPH_OP(Dequantize, ngraph::op)
NGRAPH_OP(Divide, ngraph::op)
NGRAPH_OP(Dot, ngraph::op)
NGRAPH_OP(DynBroadcast, ngraph::op)
NGRAPH_OP(DynPad, ngraph::op)
NGRAPH_OP(DynReplaceSlice, ngraph::op)
NGRAPH_OP(DynSlice, ngraph::op)
NGRAPH_OP(Elu, ngraph::op)
NGRAPH_OP(EmbeddingLookup, ngraph::op)
NGRAPH_OP(Equal, ngraph::op)
NGRAPH_OP(Erf, ngraph::op)
NGRAPH_OP(Exp, ngraph::op)
NGRAPH_OP(FakeQuantize, ngraph::op)
NGRAPH_OP(Floor, ngraph::op)
NGRAPH_OP(GRN, ngraph::op)
NGRAPH_OP(GRUCell, ngraph::op)
NGRAPH_OP(Gather, ngraph::op)
NGRAPH_OP(GatherND, ngraph::op)
NGRAPH_OP(Gelu, ngraph::op)
NGRAPH_OP(GeluBackpropFactor, ngraph::op)
NGRAPH_OP(Gemm, ngraph::op)
NGRAPH_OP(GenerateMask, ngraph::op)
NGRAPH_OP(GetOutputElement, ngraph::op)
NGRAPH_OP(Greater, ngraph::op)
NGRAPH_OP(GreaterEq, ngraph::op)
NGRAPH_OP(GroupConvolution, ngraph::op)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op)
NGRAPH_OP(GroupConvolutionBackpropFilters, ngraph::op)
NGRAPH_OP(HardSigmoid, ngraph::op)
NGRAPH_OP(Interpolate, ngraph::op)
NGRAPH_OP(LayerNorm, ngraph::op)
NGRAPH_OP(LayerNormBackprop, ngraph::op)
NGRAPH_OP(Less, ngraph::op)
NGRAPH_OP(LessEq, ngraph::op)
NGRAPH_OP(Log, ngraph::op)
NGRAPH_OP(LRN, ngraph::op)
NGRAPH_OP(LSTMCell, ngraph::op)
NGRAPH_OP(LSTMSequence, ngraph::op)
NGRAPH_OP(MatMul, ngraph::op)
NGRAPH_OP(NormalizeL2, ngraph::op)
NGRAPH_OP(Max, ngraph::op)
NGRAPH_OP(Maximum, ngraph::op)
NGRAPH_OP(MaxPool, ngraph::op)
NGRAPH_OP(MaxPoolBackprop, ngraph::op)
NGRAPH_OP(Min, ngraph::op)
NGRAPH_OP(Minimum, ngraph::op)
NGRAPH_OP(Multiply, ngraph::op)
NGRAPH_OP(MVN, ngraph::op)
NGRAPH_OP(Negative, ngraph::op)
NGRAPH_OP(Not, ngraph::op)
NGRAPH_OP(NotEqual, ngraph::op)
NGRAPH_OP(OneHot, ngraph::op)
NGRAPH_OP(Or, ngraph::op)
NGRAPH_OP(Pad, ngraph::op)
NGRAPH_OP(Parameter, ngraph::op)
NGRAPH_OP(PartialSlice, ngraph::op)
NGRAPH_OP(PartialSliceBackprop, ngraph::op)
NGRAPH_OP(Passthrough, ngraph::op)
NGRAPH_OP(Power, ngraph::op)
NGRAPH_OP(PRelu, ngraph::op)
NGRAPH_OP(Product, ngraph::op)
NGRAPH_OP(Quantize, ngraph::op)
NGRAPH_OP(QuantizedConvolution, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBias, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBiasAdd, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBiasSignedAdd, ngraph::op)
NGRAPH_OP(QuantizedConvolutionRelu, ngraph::op)
NGRAPH_OP(QuantizedDot, ngraph::op)
NGRAPH_OP(QuantizedDotBias, ngraph::op)
NGRAPH_OP(RandomUniform, ngraph::op)
NGRAPH_OP(Recv, ngraph::op)
NGRAPH_OP(Range, ngraph::op)
NGRAPH_OP(Relu, ngraph::op)
NGRAPH_OP(ReluBackprop, ngraph::op)
NGRAPH_OP(ReplaceSlice, ngraph::op)
NGRAPH_OP(Reshape, ngraph::op)
NGRAPH_OP(Result, ngraph::op)
NGRAPH_OP(Reverse, ngraph::op)
NGRAPH_OP(ReverseSequence, ngraph::op)
NGRAPH_OP(RNNCell, ngraph::op)
NGRAPH_OP(Round, ngraph::op)
NGRAPH_OP(ScalarConstantLike, ngraph::op)
NGRAPH_OP(ScaleShift, ngraph::op)
NGRAPH_OP(ScatterAdd, ngraph::op)
NGRAPH_OP(ScatterND, ngraph::op)
NGRAPH_OP(ScatterNDAdd, ngraph::op)
NGRAPH_OP(Select, ngraph::op)
NGRAPH_OP(Selu, ngraph::op)
NGRAPH_OP(Send, ngraph::op)
NGRAPH_OP(ShapeOf, ngraph::op)
NGRAPH_OP(ShuffleChannels, ngraph::op)
NGRAPH_OP(Sigmoid, ngraph::op)
NGRAPH_OP(SigmoidBackprop, ngraph::op)
NGRAPH_OP(Sign, ngraph::op)
NGRAPH_OP(Sin, ngraph::op)
NGRAPH_OP(Sinh, ngraph::op)
NGRAPH_OP(Slice, ngraph::op)
NGRAPH_OP(Softmax, ngraph::op)
NGRAPH_OP(SoftmaxCrossEntropy, ngraph::op)
NGRAPH_OP(SoftmaxCrossEntropyBackprop, ngraph::op)
NGRAPH_OP(SpaceToDepth, ngraph::op)
NGRAPH_OP(Split, ngraph::op)
NGRAPH_OP(Sqrt, ngraph::op)
NGRAPH_OP(SquaredDifference, ngraph::op)
NGRAPH_OP(Squeeze, ngraph::op)
NGRAPH_OP(StopGradient, ngraph::op)
NGRAPH_OP(Stack, ngraph::op)
NGRAPH_OP(Subtract, ngraph::op)
NGRAPH_OP(Sum, ngraph::op)
NGRAPH_OP(Tan, ngraph::op)
NGRAPH_OP(Tanh, ngraph::op)
NGRAPH_OP(TensorIterator, ngraph::op)
NGRAPH_OP(Tile, ngraph::op)
NGRAPH_OP(TopK, ngraph::op::v0)
NGRAPH_OP(Unsqueeze, ngraph::op)
NGRAPH_OP(Xor, ngraph::op)
