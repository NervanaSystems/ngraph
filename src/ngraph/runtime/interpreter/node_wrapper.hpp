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

#pragma once

#include <memory>

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            enum class OP_TYPEID;
            class NodeWrapper;
        }
    }
}

enum class ngraph::runtime::interpreter::OP_TYPEID
{
    Abs_TYPEID,
    Acos_TYPEID,
    Add_TYPEID,
    AllReduce_TYPEID,
    And_TYPEID,
    ArgMax_TYPEID,
    ArgMin_TYPEID,
    Asin_TYPEID,
    Atan_TYPEID,
    AvgPool_TYPEID,
    AvgPoolBackprop_TYPEID,
    BatchNorm_TYPEID,
    BatchNormBackprop_TYPEID,
    Broadcast_TYPEID,
    Ceiling_TYPEID,
    Concat_TYPEID,
    Constant_TYPEID,
    Convert_TYPEID,
    Convolution_TYPEID,
    ConvolutionBackpropData_TYPEID,
    ConvolutionBackpropFilters_TYPEID,
    Cos_TYPEID,
    Cosh_TYPEID,
    Divide_TYPEID,
    Dot_TYPEID,
    Equal_TYPEID,
    Exp_TYPEID,
    Floor_TYPEID,
    FunctionCall_TYPEID,
    GetOutputElement_TYPEID,
    Greater_TYPEID,
    GreaterEq_TYPEID,
    Less_TYPEID,
    LessEq_TYPEID,
    Log_TYPEID,
    LRN_TYPEID,
    Max_TYPEID,
    Maximum_TYPEID,
    MaxPool_TYPEID,
    MaxPoolBackprop_TYPEID,
    Min_TYPEID,
    Minimum_TYPEID,
    Multiply_TYPEID,
    Negative_TYPEID,
    Not_TYPEID,
    NotEqual_TYPEID,
    OneHot_TYPEID,
    Or_TYPEID,
    Pad_TYPEID,
    Parameter_TYPEID,
    Power_TYPEID,
    Product_TYPEID,
    Reduce_TYPEID,
    ReduceWindow_TYPEID,
    Relu_TYPEID,
    ReluBackprop_TYPEID,
    ReplaceSlice_TYPEID,
    Reshape_TYPEID,
    Result_TYPEID,
    Reverse_TYPEID,
    ReverseSequence_TYPEID,
    Select_TYPEID,
    SelectAndScatter_TYPEID,
    Sigmoid_TYPEID,
    SigmoidBackprop_TYPEID,
    Sign_TYPEID,
    Sin_TYPEID,
    Sinh_TYPEID,
    Slice_TYPEID,
    Softmax_TYPEID,
    Sqrt_TYPEID,
    StopGradient_TYPEID,
    Subtract_TYPEID,
    Sum_TYPEID,
    Tan_TYPEID,
    Tanh_TYPEID
};

class ngraph::runtime::interpreter::NodeWrapper
{
public:
    NodeWrapper(const std::shared_ptr<ngraph::Node>& node,
                ngraph::runtime::interpreter::OP_TYPEID tid)
        : m_node{node}
        , m_typeid{tid}
    {
    }

    Node& get_node() const { return *m_node; }
    ngraph::runtime::interpreter::OP_TYPEID get_typeid() const { return m_typeid; }
private:
    std::shared_ptr<ngraph::Node> m_node;
    OP_TYPEID m_typeid;
};
