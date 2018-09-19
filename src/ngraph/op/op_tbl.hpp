//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
// #define NGRAPH_OP(a) #a,
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
// It's that easy. You can use this for fun and profit.

NGRAPH_OP(Abs)
NGRAPH_OP(Acos)
NGRAPH_OP(Add)
NGRAPH_OP(AllReduce)
NGRAPH_OP(And)
NGRAPH_OP(ArgMax)
NGRAPH_OP(ArgMin)
NGRAPH_OP(Asin)
NGRAPH_OP(Atan)
NGRAPH_OP(AvgPool)
NGRAPH_OP(AvgPoolBackprop)
NGRAPH_OP(BatchNorm)
NGRAPH_OP(BatchNormBackprop)
NGRAPH_OP(Broadcast)
NGRAPH_OP(Ceiling)
NGRAPH_OP(Concat)
NGRAPH_OP(Constant)
NGRAPH_OP(Convert)
NGRAPH_OP(Convolution)
NGRAPH_OP(ConvolutionBackpropData)
NGRAPH_OP(ConvolutionBackpropFilters)
NGRAPH_OP(Cos)
NGRAPH_OP(Cosh)
NGRAPH_OP(Divide)
NGRAPH_OP(Dot)
NGRAPH_OP(Equal)
NGRAPH_OP(Exp)
NGRAPH_OP(Floor)
NGRAPH_OP(FunctionCall)
NGRAPH_OP(GetOutputElement)
NGRAPH_OP(Greater)
NGRAPH_OP(GreaterEq)
NGRAPH_OP(Less)
NGRAPH_OP(LessEq)
NGRAPH_OP(Log)
NGRAPH_OP(LRN)
NGRAPH_OP(Max)
NGRAPH_OP(Maximum)
NGRAPH_OP(MaxPool)
NGRAPH_OP(MaxPoolBackprop)
NGRAPH_OP(Min)
NGRAPH_OP(Minimum)
NGRAPH_OP(Multiply)
NGRAPH_OP(Negative)
NGRAPH_OP(Not)
NGRAPH_OP(NotEqual)
NGRAPH_OP(OneHot)
NGRAPH_OP(Or)
NGRAPH_OP(Pad)
NGRAPH_OP(Parameter)
NGRAPH_OP(Power)
NGRAPH_OP(Product)
NGRAPH_OP(Reduce)
NGRAPH_OP(ReduceWindow)
NGRAPH_OP(Relu)
NGRAPH_OP(ReluBackprop)
NGRAPH_OP(ReplaceSlice)
NGRAPH_OP(Reshape)
NGRAPH_OP(Result)
NGRAPH_OP(Reverse)
NGRAPH_OP(ReverseSequence)
NGRAPH_OP(Select)
NGRAPH_OP(SelectAndScatter)
NGRAPH_OP(Sigmoid)
NGRAPH_OP(SigmoidBackprop)
NGRAPH_OP(Sign)
NGRAPH_OP(Sin)
NGRAPH_OP(Sinh)
NGRAPH_OP(Slice)
NGRAPH_OP(Softmax)
NGRAPH_OP(Sqrt)
NGRAPH_OP(StopGradient)
NGRAPH_OP(Subtract)
NGRAPH_OP(Sum)
NGRAPH_OP(Tan)
NGRAPH_OP(Tanh)
NGRAPH_OP(TopK)
