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

#pragma once

#include "ngraph/ngraph.hpp"

namespace ngraph
{
    namespace opset0
    {
        using op::v0::Abs;
        using op::v0::Acos;
        using op::v0::Add;
        using op::v0::All;
        using op::v0::AllReduce;
        using op::v0::And;
        using op::v0::ArgMax;
        using op::v0::ArgMin;
        using op::v0::Asin;
        using op::v0::Atan;
        using op::v0::AvgPool;
        using op::v0::BatchNormInference;
        using op::v0::BatchNormTraining;
        using op::v0::BatchNormTrainingBackprop;
        using op::v0::Broadcast;
        using op::v0::BroadcastDistributed;
        using op::v0::Ceiling;
        using op::v0::Concat;
        // TODO: using op::v0::Constant;
        using op::v0::Convert;
        using op::v0::Convolution;
        using op::v0::ConvolutionBackpropData;
        using op::v0::ConvolutionBackpropFilters;
        using op::v0::Cos;
        using op::v0::Cosh;
        using op::v0::Divide;
        using op::v0::Dot;
        using op::v0::EmbeddingLookup;
        using op::v0::Equal;
        using op::v0::Erf;
        using op::v0::Exp;
        using op::v0::Floor;
        using op::v0::Gather;
        using op::v0::GatherND;
        using op::v0::Greater;
        using op::v0::GreaterEq;
        using op::v0::LessEq;
        using op::v0::Less;
        using op::v0::Log;
        using op::v0::LRN;
        using op::v0::Max;
        using op::v0::MaxPool;
        using op::v0::MaxPoolBackprop;
        using op::v0::Maximum;
        using op::v0::Min;
        using op::v0::Minimum;
        using op::v0::Multiply;
        using op::v0::Negative;
        using op::v0::Not;
        using op::v0::NotEqual;
        using op::v0::OneHot;
        using op::v0::Or;
        using op::v0::Pad;
        // TODO: using op::v0::Parameter;
        using op::v0::Passthrough;
        using op::v0::Product;
        using op::v0::Power;
        using op::v0::Quantize;
        using op::v0::QuantizedConvolution;
        using op::v0::QuantizedDot;
        using op::v0::Recv;
        using op::v0::Relu;
        using op::v0::ReplaceSlice;
        using op::v0::Reshape;
        // TODO: using op::v0::Result;
        using op::v0::Reverse;
        using op::v0::ReverseSequence;
        using op::v0::ScatterAdd;
        using op::v0::ScatterNDAdd;
        using op::v0::Select;
        using op::v0::Send;
        using op::v0::Sigmoid;
        using op::v0::SigmoidBackprop;
        using op::v0::Sign;
        using op::v0::Sin;
        using op::v0::Sinh;
        using op::v0::Softmax;
        using op::v0::Sqrt;
        using op::v0::StopGradient;
        using op::v0::Subtract;
        using op::v0::Sum;
        using op::v0::Tan;
        using op::v0::Tanh;
        using op::v0::TopK;
        using op::v0::Xor;
    }
}