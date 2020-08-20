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

namespace
{
    void op_is_Abs()
    {
        op::v0::Abs node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Acos()
    {
        op::v0::Acos node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Add()
    {
        op::v1::Add node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_All()
    {
        op::v0::All node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_AllReduce()
    {
        op::v0::AllReduce node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Any()
    {
        op::v0::Any node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ArgMax()
    {
        op::v0::ArgMax node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ArgMin()
    {
        op::v0::ArgMin node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Asin()
    {
        op::v0::Asin node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Atan()
    {
        op::v0::Atan node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Atan2()
    {
        op::v0::Atan2 node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_AvgPool()
    {
        op::v0::AvgPool node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_AvgPoolBackprop()
    {
        op::v0::AvgPoolBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchMatMul()
    {
        op::v0::BatchMatMul node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchMatMulTranspose()
    {
        op::v0::BatchMatMulTranspose node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchNormInference()
    {
        op::v0::BatchNormInference node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchNormTraining()
    {
        op::v0::BatchNormTraining node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchNormTrainingBackprop()
    {
        op::v0::BatchNormTrainingBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Broadcast()
    {
        op::v0::Broadcast node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BroadcastDistributed()
    {
        op::v0::BroadcastDistributed node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BroadcastLike()
    {
        op::v0::BroadcastLike node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Ceiling()
    {
        op::v0::Ceiling node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Clamp()
    {
        op::v0::Clamp node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Concat()
    {
        op::v0::Concat node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Constant()
    {
        op::v0::Constant node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Convert()
    {
        op::v0::Convert node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Convolution()
    {
        op::v0::Convolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBackpropData()
    {
        op::v0::ConvolutionBackpropData node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBackpropFilters()
    {
        op::v0::ConvolutionBackpropFilters node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBias()
    {
        op::v0::ConvolutionBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBiasAdd()
    {
        op::v0::ConvolutionBiasAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBiasBackpropFiltersBias()
    {
        op::v0::ConvolutionBiasBackpropFiltersBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Cos()
    {
        op::v0::Cos node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Cosh()
    {
        op::v0::Cosh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CrossEntropy()
    {
        op::v0::CrossEntropy node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CrossEntropyBackprop()
    {
        op::v0::CrossEntropyBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CropAndResize()
    {
        op::v0::CropAndResize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CumSum()
    {
        op::v0::CumSum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DepthToSpace()
    {
        op::v0::DepthToSpace node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Dequantize()
    {
        op::v0::Dequantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Divide()
    {
        op::v1::Divide node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Dot()
    {
        op::v0::Dot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynBroadcast()
    {
        op::v0::DynBroadcast node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynPad()
    {
        op::v0::DynPad node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynReplaceSlice()
    {
        op::v0::DynReplaceSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynSlice()
    {
        op::v0::DynSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Elu()
    {
        op::v0::Elu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_EmbeddingLookup()
    {
        op::v0::EmbeddingLookup node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Equal()
    {
        op::v1::Equal node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Erf()
    {
        op::v0::Erf node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Exp()
    {
        op::v0::Exp node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_FakeQuantize()
    {
        op::v0::FakeQuantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Floor()
    {
        op::v0::Floor node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GRN()
    {
        op::v0::GRN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GRUCell()
    {
        op::v3::GRUCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gather()
    {
        op::v0::Gather node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GatherND()
    {
        op::v0::GatherND node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gelu()
    {
        op::v0::Gelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GeluBackpropFactor()
    {
        op::v0::GeluBackpropFactor node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gemm()
    {
        op::v0::Gemm node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GenerateMask()
    {
        op::v0::GenerateMask node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Greater()
    {
        op::v1::Greater node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GreaterEqual()
    {
        op::v1::GreaterEqual node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GroupConvolution()
    {
        op::v0::GroupConvolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GroupConvolutionBackpropData()
    {
        op::v0::GroupConvolutionBackpropData node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GroupConvolutionBackpropFilters()
    {
        op::v0::GroupConvolutionBackpropFilters node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_HardSigmoid()
    {
        op::v0::HardSigmoid node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Interpolate()
    {
        op::v0::Interpolate node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LayerNorm()
    {
        op::v0::LayerNorm node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LayerNormBackprop()
    {
        op::v0::LayerNormBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Less()
    {
        op::v1::Less node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LessEqual()
    {
        op::v1::LessEqual node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Log()
    {
        op::v0::Log node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LogicalAnd()
    {
        op::v1::LogicalAnd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }

    void op_is_LogicalNot()
    {
        op::v1::LogicalNot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LogicalOr()
    {
        op::v1::LogicalOr node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }

    void op_is_LogicalXor()
    {
        op::v1::LogicalXor node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }

    void op_is_LRN()
    {
        op::v0::LRN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LSTMCell()
    {
        op::v0::LSTMCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LSTMSequence()
    {
        op::v0::LSTMSequence node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MatMul()
    {
        op::v0::MatMul node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_NormalizeL2()
    {
        op::v0::NormalizeL2 node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Max()
    {
        op::v0::Max node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Maximum()
    {
        op::v1::Maximum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MaxPool()
    {
        op::v0::MaxPool node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MaxPoolBackprop()
    {
        op::v0::MaxPoolBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Min()
    {
        op::v0::Min node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Minimum()
    {
        op::v1::Minimum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Multiply()
    {
        op::v1::Multiply node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MVN()
    {
        op::v0::MVN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Negative()
    {
        op::v0::Negative node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_NotEqual()
    {
        op::v1::NotEqual node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_OneHot()
    {
        op::v0::OneHot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Pad()
    {
        op::v0::Pad node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Parameter()
    {
        op::v0::Parameter node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_PartialSlice()
    {
        op::v0::PartialSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_PartialSliceBackprop()
    {
        op::v0::PartialSliceBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Passthrough()
    {
        op::v0::Passthrough node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Power()
    {
        op::v1::Power node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_PRelu()
    {
        op::v0::PRelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Product()
    {
        op::v0::Product node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Quantize()
    {
        op::v0::Quantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolution()
    {
        op::v0::QuantizedConvolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBias()
    {
        op::v0::QuantizedConvolutionBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBiasAdd()
    {
        op::v0::QuantizedConvolutionBiasAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBiasSignedAdd()
    {
        op::v0::QuantizedConvolutionBiasSignedAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionRelu()
    {
        op::v0::QuantizedConvolutionRelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedDot()
    {
        op::v0::QuantizedDot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedDotBias()
    {
        op::v0::QuantizedDotBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_RandomUniform()
    {
        op::v0::RandomUniform node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Recv()
    {
        op::v0::Recv node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Range()
    {
        op::v0::Range node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Relu()
    {
        op::v0::Relu node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ReluBackprop()
    {
        op::v0::ReluBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ReplaceSlice()
    {
        op::v0::ReplaceSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Reshape()
    {
        op::v0::Reshape node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Result()
    {
        op::v0::Result node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Reverse()
    {
        op::v0::Reverse node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ReverseSequence()
    {
        op::v0::ReverseSequence node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_RNNCell()
    {
        op::v0::RNNCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Round()
    {
        op::v0::Round node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScalarConstantLike()
    {
        op::v0::ScalarConstantLike node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScaleShift()
    {
        op::v0::ScaleShift node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterAdd()
    {
        op::v0::ScatterAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterND()
    {
        op::v0::ScatterND node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterNDAdd()
    {
        op::v0::ScatterNDAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Select()
    {
        op::v0::Select node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Selu()
    {
        op::v0::Selu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Send()
    {
        op::v0::Send node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ShapeOf()
    {
        op::v0::ShapeOf node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ShuffleChannels()
    {
        op::v0::ShuffleChannels node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sigmoid()
    {
        op::v0::Sigmoid node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SigmoidBackprop()
    {
        op::v0::SigmoidBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sign()
    {
        op::v0::Sign node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sin()
    {
        op::v0::Sin node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sinh()
    {
        op::v0::Sinh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Slice()
    {
        op::v0::Slice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Softmax()
    {
        op::v0::Softmax node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SoftmaxCrossEntropy()
    {
        op::v0::SoftmaxCrossEntropy node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SoftmaxCrossEntropyBackprop()
    {
        op::v0::SoftmaxCrossEntropyBackprop node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SpaceToDepth()
    {
        op::v0::SpaceToDepth node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Split()
    {
        op::v0::Split node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sqrt()
    {
        op::v0::Sqrt node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SquaredDifference()
    {
        op::v0::SquaredDifference node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Squeeze()
    {
        op::v0::Squeeze node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_StopGradient()
    {
        op::v0::StopGradient node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Stack()
    {
        op::v0::Stack node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Subtract()
    {
        op::v1::Subtract node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sum()
    {
        op::v0::Sum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tan()
    {
        op::v0::Tan node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tanh()
    {
        op::v0::Tanh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_TensorIterator()
    {
        op::v0::TensorIterator node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tile()
    {
        op::v0::Tile node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_TopK()
    {
        op::v0::TopK node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Unsqueeze()
    {
        op::v0::Unsqueeze node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }
}

TEST(op_is, check)
{
#define NGRAPH_OP(a, b) op_is_##a();
#include "ngraph/opset/opset0_tbl.hpp"
#undef NGRAPH_OP
}
