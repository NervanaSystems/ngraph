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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

TEST(op_is, Abs)
{
    op::Abs node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Acos)
{
    op::Acos node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Add)
{
    op::Add node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, All)
{
    op::All node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, AllReduce)
{
    op::AllReduce node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, And)
{
    op::And node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_TRUE(node.is_binary_elementwise_logical());
}

TEST(op_is, Any)
{
    op::Any node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ArgMax)
{
    op::ArgMax node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ArgMin)
{
    op::ArgMin node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Asin)
{
    op::Asin node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Atan)
{
    op::Atan node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Atan2)
{
    op::Atan2 node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, AvgPool)
{
    op::AvgPool node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, AvgPoolBackprop)
{
    op::AvgPoolBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BatchMatMul)
{
    op::BatchMatMul node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BatchMatMulTranspose)
{
    op::BatchMatMulTranspose node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BatchNormInference)
{
    op::BatchNormInference node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BatchNormTraining)
{
    op::BatchNormTraining node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BatchNormTrainingBackprop)
{
    op::BatchNormTrainingBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Broadcast)
{
    op::Broadcast node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BroadcastDistributed)
{
    op::BroadcastDistributed node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, BroadcastLike)
{
    op::BroadcastLike node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Ceiling)
{
    op::Ceiling node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Clamp)
{
    op::Clamp node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Concat)
{
    op::Concat node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Constant)
{
    op::Constant node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Convert)
{
    op::Convert node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Convolution)
{
    op::Convolution node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ConvolutionBackpropData)
{
    op::ConvolutionBackpropData node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ConvolutionBackpropFilters)
{
    op::ConvolutionBackpropFilters node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ConvolutionBias)
{
    op::ConvolutionBias node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ConvolutionBiasAdd)
{
    op::ConvolutionBiasAdd node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ConvolutionBiasBackpropFiltersBias)
{
    op::ConvolutionBiasBackpropFiltersBias node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Cos)
{
    op::Cos node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Cosh)
{
    op::Cosh node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, CrossEntropy)
{
    op::CrossEntropy node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, CrossEntropyBackprop)
{
    op::CrossEntropyBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, CropAndResize)
{
    op::CropAndResize node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, CumSum)
{
    op::CumSum node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DepthToSpace)
{
    op::DepthToSpace node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Dequantize)
{
    op::Dequantize node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Divide)
{
    op::Divide node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Dot)
{
    op::Dot node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DynBroadcast)
{
    op::DynBroadcast node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DynPad)
{
    op::DynPad node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DynReplaceSlice)
{
    op::DynReplaceSlice node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DynReshape)
{
    op::DynReshape node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, DynSlice)
{
    op::DynSlice node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Elu)
{
    op::Elu node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, EmbeddingLookup)
{
    op::EmbeddingLookup node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Equal)
{
    op::Equal node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Erf)
{
    op::Erf node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Exp)
{
    op::Exp node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, FakeQuantize)
{
    op::FakeQuantize node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Floor)
{
    op::Floor node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GRN)
{
    op::GRN node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GRUCell)
{
    op::GRUCell node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Gather)
{
    op::Gather node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GatherND)
{
    op::GatherND node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Gelu)
{
    op::Gelu node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GeluBackpropFactor)
{
    op::GeluBackpropFactor node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Gemm)
{
    op::Gemm node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GenerateMask)
{
    op::GenerateMask node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GetOutputElement)
{
    op::GetOutputElement node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Greater)
{
    op::Greater node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GreaterEq)
{
    op::GreaterEq node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GroupConvolution)
{
    op::GroupConvolution node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GroupConvolutionBackpropData)
{
    op::GroupConvolutionBackpropData node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GroupConvolutionBackpropFilters)
{
    op::GroupConvolutionBackpropFilters node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, GroupConvolutionTranspose)
{
    op::GroupConvolutionTranspose node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, HardSigmoid)
{
    op::HardSigmoid node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Interpolate)
{
    op::Interpolate node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LayerNorm)
{
    op::LayerNorm node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LayerNormBackprop)
{
    op::LayerNormBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Less)
{
    op::Less node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LessEq)
{
    op::LessEq node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Log)
{
    op::Log node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LogSoftmax)
{
    op::LogSoftmax node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LRN)
{
    op::LRN node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LSTMCell)
{
    op::LSTMCell node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, LSTMSequence)
{
    op::LSTMSequence node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, MatMul)
{
    op::MatMul node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, NormalizeL2)
{
    op::NormalizeL2 node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Max)
{
    op::Max node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Maximum)
{
    op::Maximum node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, MaxPool)
{
    op::MaxPool node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, MaxPoolBackprop)
{
    op::MaxPoolBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Min)
{
    op::Min node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Minimum)
{
    op::Minimum node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Multiply)
{
    op::Multiply node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, MVN)
{
    op::MVN node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Negative)
{
    op::Negative node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Not)
{
    op::Not node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, NotEqual)
{
    op::NotEqual node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, OneHot)
{
    op::OneHot node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Or)
{
    op::Or node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_TRUE(node.is_binary_elementwise_logical());
}

TEST(op_is, Pad)
{
    op::Pad node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Parameter)
{
    op::Parameter node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, PartialSlice)
{
    op::PartialSlice node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, PartialSliceBackprop)
{
    op::PartialSliceBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Passthrough)
{
    op::Passthrough node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Power)
{
    op::Power node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, PRelu)
{
    op::PRelu node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Product)
{
    op::Product node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Quantize)
{
    op::Quantize node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedConvolution)
{
    op::QuantizedConvolution node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedConvolutionBias)
{
    op::QuantizedConvolutionBias node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedConvolutionBiasAdd)
{
    op::QuantizedConvolutionBiasAdd node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedConvolutionBiasSignedAdd)
{
    op::QuantizedConvolutionBiasSignedAdd node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedConvolutionRelu)
{
    op::QuantizedConvolutionRelu node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedDot)
{
    op::QuantizedDot node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, QuantizedDotBias)
{
    op::QuantizedDotBias node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, RandomUniform)
{
    op::RandomUniform node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Recv)
{
    op::Recv node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Range)
{
    op::Range node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Reciprocal)
{
    op::Reciprocal node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Relu)
{
    op::Relu node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ReluBackprop)
{
    op::ReluBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ReplaceSlice)
{
    op::ReplaceSlice node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Reshape)
{
    op::Reshape node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Result)
{
    op::Result node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Reverse)
{
    op::Reverse node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ReverseSequence)
{
    op::ReverseSequence node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, RNNCell)
{
    op::RNNCell node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ScalarConstantLike)
{
    op::ScalarConstantLike node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ScaleShift)
{
    op::ScaleShift node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ScatterAdd)
{
    op::ScatterAdd node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ScatterND)
{
    op::ScatterND node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ScatterNDAdd)
{
    op::ScatterNDAdd node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Select)
{
    op::Select node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Selu)
{
    op::Selu node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Send)
{
    op::Send node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ShapeOf)
{
    op::ShapeOf node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, ShuffleChannels)
{
    op::ShuffleChannels node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sigmoid)
{
    op::Sigmoid node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, SigmoidBackprop)
{
    op::SigmoidBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sign)
{
    op::Sign node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sin)
{
    op::Sin node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sinh)
{
    op::Sinh node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Slice)
{
    op::Slice node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Softmax)
{
    op::Softmax node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, SoftmaxCrossEntropy)
{
    op::SoftmaxCrossEntropy node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, SoftmaxCrossEntropyBackprop)
{
    op::SoftmaxCrossEntropyBackprop node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, SpaceToDepth)
{
    op::SpaceToDepth node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Split)
{
    op::Split node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sqrt)
{
    op::Sqrt node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, SquaredDifference)
{
    op::SquaredDifference node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Squeeze)
{
    op::Squeeze node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, StopGradient)
{
    op::StopGradient node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Stack)
{
    op::Stack node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Subtract)
{
    op::Subtract node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Sum)
{
    op::Sum node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Tan)
{
    op::Tan node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Tanh)
{
    op::Tanh node;
    EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, TensorIterator)
{
    op::TensorIterator node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Tile)
{
    op::Tile node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, TopK)
{
    op::TopK node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Transpose)
{
    op::Transpose node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Unsqueeze)
{
    op::Unsqueeze node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_FALSE(node.is_binary_elementwise_logical());
}

TEST(op_is, Xor)
{
    op::Xor node;
    EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
    EXPECT_FALSE(node.is_binary_elementwise_comparison());
    EXPECT_TRUE(node.is_binary_elementwise_logical());
}

TEST(op_is, check)
{
#define NGRAPH_OP(a, b) run_test_ ## a();
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
}
