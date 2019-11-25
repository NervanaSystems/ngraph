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

NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(AvgPool, ngraph::op::v1)
NGRAPH_OP(AvgPoolBackprop, ngraph::op::v1)
NGRAPH_OP(BinaryConvolution, ngraph::op::v1)
NGRAPH_OP(Broadcast, ngraph::op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropFilters, ngraph::op::v1)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Gather, ngraph::op::v1)
NGRAPH_OP(GenerateMask, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEqual, ngraph::op::v1)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(MaxPool, ngraph::op::v1)
NGRAPH_OP(MaxPoolBackprop, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Multiply, ngraph::op::v1)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(OneHot, ngraph::op::v1)
// NGRAPH_OP(PSROIPooling, ngraph::op)
NGRAPH_OP(Pad, ngraph::op::v1)
NGRAPH_OP(Power, ngraph::op::v1)
NGRAPH_OP(ReduceMax, ngraph::op::v1)
NGRAPH_OP(ReduceMean, ngraph::op::v1)
NGRAPH_OP(ReduceMin, ngraph::op::v1)
NGRAPH_OP(ReduceProd, ngraph::op::v1)
NGRAPH_OP(ReduceSum, ngraph::op::v1)
NGRAPH_OP(Reshape, ngraph::op::v1)
NGRAPH_OP(Reverse, ngraph::op::v1)
NGRAPH_OP(Softmax, ngraph::op::v1)
NGRAPH_OP(StridedSlice, ngraph::op::v1)
NGRAPH_OP(TopK, ngraph::op::v1)
NGRAPH_OP(VariadicSplit, ngraph::op::v1)
