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
    namespace opset1
    {
        using op::v0::Abs;
        using op::v0::Acos;
        // TODO: using op::v0::Acosh;
        using op::v1::Add;
        using op::v0::Asin;
        // TODO: using op::v0::Asinh;
        using op::v1::LogicalAnd;
        using op::v0::Atan;
        // TODO: using op::v0::Atanh;
        using op::v1::AvgPool;
        using op::v0::BatchNormInference;
        using op::v1::Broadcast;
        using op::v0::Ceiling;
        using op::v0::Concat;
        // TODO: using op::v0::Constant;
        using op::v0::Convert;
        // TODO: using op::v0::ConvertLike;
        using op::v1::Convolution;
        using op::v1::ConvolutionBackpropData;
        using op::v0::Cos;
        using op::v0::Cosh;
        using op::v0::CTCGreedyDecoder;
        // TODO: using op::v0::DeformableConvolution
        // TODO: using op::v0::DeformablePSROIPooling
        using op::v0::DepthToSpace;
        using op::v0::DetectionOutput;
        using op::v1::Divide;
        using op::v0::Elu;
        using op::v1::Equal;
        using op::v0::Erf;
        using op::v0::Exp;
        using op::v0::FakeQuantize;
        using op::v0::Floor;
        // TODO: using op::v0::FloorMod;
        using op::v1::Gather;
        // TODO: using op::v0::GatherTree;
        using op::v0::Greater;
        using op::v0::GreaterEq;
        using op::v0::GRN;
        using op::v0::GroupConvolution;
        using op::v0::GroupConvolutionTranspose;
        using op::v0::GRUCell;
        // TODO using op::v0::GRUSequence;
        using op::v0::HardSigmoid;
        using op::v0::Interpolate;
        using op::v0::Less;
        using op::v0::LessEq;
        using op::v0::Log;
        using op::v1::LogicalAnd;
        using op::v1::LogicalNot;
        using op::v1::LogicalOr;
        using op::v1::LogicalXor;
        using op::v0::LRN;
        using op::v0::MatMul;
        using op::v0::Maximum;
        using op::v1::MaxPool;
        using op::v1::Minimum;
        // TODO using op::v0::Mod;
        using op::v1::Multiply;
        using op::v0::Negative;
        // TODO using op::v0::NonMaxSuppression
        using op::v0::NormalizeL2;
        using op::v1::NotEqual;
        // TODO using op::v1::OneHot;
        using op::v1::Pad;
        // TODO: using op::v0::Parameter;
        using op::v0::Power;
        using op::v0::PRelu;
        using op::v0::PriorBox;
        using op::v0::PriorBoxClustered;
        using op::v0::Proposal;
        using op::v0::PSROIPooling;
        // TODO using op::v0::ReduceLogicalAnd;
        // TODO using op::v0::ReduceLogicalOr;
        using op::v1::ReduceMax;
        using op::v1::ReduceMean;
        using op::v1::ReduceMin;
        using op::v1::ReduceProd;
        using op::v1::ReduceSum;
        using op::v0::RegionYolo;
        using op::v0::Relu;
        using op::v1::Reshape;
        // TODO: using op::v0::Result;
        using op::v0::Reverse;
        using op::v0::ReverseSequence;
        using op::v0::RNNCell;
        using op::v0::Select;
        using op::v0::Selu;
        using op::v0::ShapeOf;
        using op::v0::ShuffleChannels;
        using op::v0::Sigmoid;
        using op::v0::Sign;
        using op::v0::Sin;
        using op::v0::Sinh;
        using op::v1::Softmax;
        using op::v0::SpaceToDepth;
        using op::v0::Split;
        using op::v0::Sqrt;
        using op::v0::SquaredDifference;
        using op::v0::Squeeze;
        using op::v1::StridedSlice;
        using op::v0::Subtract;
        using op::v0::Tan;
        using op::v0::Tanh;
        using op::v0::TensorIterator;
        using op::v0::Tile;
        using op::v1::TopK;
        using op::v0::Transpose;
        using op::v0::Unsqueeze;
        // TODO using op::v0::VariadicSplit
        using op::v0::Xor;
    }
}