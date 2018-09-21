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

#include <algorithm>
#include <functional>

#include "core/attribute.hpp"
#include "op/abs.hpp"
#include "op/add.hpp"
#include "op/and.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/cast.hpp"
#include "op/ceil.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/div.hpp"
#include "op/elu.hpp"
#include "op/equal.hpp"
#include "op/exp.hpp"
#include "op/flatten.hpp"
#include "op/floor.hpp"
#include "op/gemm.hpp"
#include "op/greater.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/identity.hpp"
#include "op/leaky_relu.hpp"
#include "op/less.hpp"
#include "op/log.hpp"
#include "op/log_softmax.hpp"
#include "op/lrn.hpp"
#include "op/matmul.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/mean.hpp"
#include "op/min.hpp"
#include "op/mul.hpp"
#include "op/neg.hpp"
#include "op/not.hpp"
#include "op/or.hpp"
#include "op/pow.hpp"
#include "op/prelu.hpp"
#include "op/reciprocal.hpp"
#include "op/reduce.hpp"
#include "op/relu.hpp"
#include "op/reshape.hpp"
#include "op/selu.hpp"
#include "op/shape.hpp"
#include "op/sigmoid.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/softplus.hpp"
#include "op/softsign.hpp"
#include "op/split.hpp"
#include "op/sqrt.hpp"
#include "op/squeeze.hpp"
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/tanh.hpp"
#include "op/thresholded_relu.hpp"
#include "op/transpose.hpp"
#include "op/unsqueeze.hpp"
#include "op/xor.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace error
            {
                struct unknown_operation : ngraph_error
                {
                    explicit unknown_operation(const std::string& op_type)
                        : ngraph_error{"unknown operation: " + op_type}
                    {
                    }
                };

            } // namespace error

            class ops_bridge
            {
            public:
                ops_bridge(const ops_bridge&) = delete;
                ops_bridge& operator=(const ops_bridge&) = delete;
                ops_bridge(ops_bridge&&) = delete;
                ops_bridge& operator=(ops_bridge&&) = delete;

                static NodeVector make_ng_nodes(const Node& node)
                {
                    return ops_bridge::get()(node);
                }

            private:
                std::map<std::string, std::function<NodeVector(const Node&)>> m_map;

                static const ops_bridge& get()
                {
                    static ops_bridge instance;
                    return instance;
                }

                ops_bridge()
                {
                    m_map.emplace("Abs", std::bind(op::abs, std::placeholders::_1));
                    m_map.emplace("Add", std::bind(op::add, std::placeholders::_1));
                    m_map.emplace("And", std::bind(op::logical_and, std::placeholders::_1));
                    m_map.emplace("AveragePool",
                                  std::bind(op::average_pool, std::placeholders::_1));
                    m_map.emplace("BatchNormalization",
                                  std::bind(op::batch_norm, std::placeholders::_1));
                    m_map.emplace("Cast", std::bind(op::cast, std::placeholders::_1));
                    m_map.emplace("Ceil", std::bind(op::ceil, std::placeholders::_1));
                    m_map.emplace("Clip", std::bind(op::clip, std::placeholders::_1));
                    m_map.emplace("Concat", std::bind(op::concat, std::placeholders::_1));
                    m_map.emplace("Constant", std::bind(op::constant, std::placeholders::_1));
                    m_map.emplace("Conv", std::bind(op::conv, std::placeholders::_1));
                    m_map.emplace("Div", std::bind(op::div, std::placeholders::_1));
                    m_map.emplace("Dropout", std::bind(op::identity, std::placeholders::_1));
                    m_map.emplace("Elu", std::bind(op::elu, std::placeholders::_1));
                    m_map.emplace("Equal", std::bind(op::equal, std::placeholders::_1));
                    m_map.emplace("Exp", std::bind(op::exp, std::placeholders::_1));
                    m_map.emplace("Flatten", std::bind(op::flatten, std::placeholders::_1));
                    m_map.emplace("Floor", std::bind(op::floor, std::placeholders::_1));
                    m_map.emplace("Gemm", std::bind(op::gemm, std::placeholders::_1));
                    m_map.emplace("Greater", std::bind(op::greater, std::placeholders::_1));
                    m_map.emplace("HardSigmoid",
                                  std::bind(op::hard_sigmoid, std::placeholders::_1));
                    m_map.emplace("Identity", std::bind(op::identity, std::placeholders::_1));
                    m_map.emplace("LeakyRelu", std::bind(op::leaky_relu, std::placeholders::_1));
                    m_map.emplace("Less", std::bind(op::less, std::placeholders::_1));
                    m_map.emplace("Log", std::bind(op::log, std::placeholders::_1));
                    m_map.emplace("LogSoftmax", std::bind(op::log_softmax, std::placeholders::_1));
                    m_map.emplace("LRN", std::bind(op::lrn, std::placeholders::_1));
                    m_map.emplace("MatMul", std::bind(op::matmul, std::placeholders::_1));
                    m_map.emplace("MaxPool", std::bind(op::max_pool, std::placeholders::_1));
                    m_map.emplace("Max", std::bind(op::max, std::placeholders::_1));
                    m_map.emplace("Mean", std::bind(op::mean, std::placeholders::_1));
                    m_map.emplace("Min", std::bind(op::min, std::placeholders::_1));
                    m_map.emplace("Mul", std::bind(op::mul, std::placeholders::_1));
                    m_map.emplace("Neg", std::bind(op::neg, std::placeholders::_1));
                    m_map.emplace("Not", std::bind(op::logical_not, std::placeholders::_1));
                    m_map.emplace("Or", std::bind(op::logical_or, std::placeholders::_1));
                    m_map.emplace("Pow", std::bind(op::pow, std::placeholders::_1));
                    m_map.emplace("PRelu", std::bind(op::prelu, std::placeholders::_1));
                    m_map.emplace("Reciprocal", std::bind(op::reciprocal, std::placeholders::_1));
                    m_map.emplace("ReduceLogSum",
                                  std::bind(op::reduce_log_sum, std::placeholders::_1));
                    m_map.emplace("ReduceLogSumExp",
                                  std::bind(op::reduce_log_sum_exp, std::placeholders::_1));
                    m_map.emplace("ReduceL1", std::bind(op::reduce_l1, std::placeholders::_1));
                    m_map.emplace("ReduceL2", std::bind(op::reduce_l2, std::placeholders::_1));
                    m_map.emplace("ReduceMax", std::bind(op::reduce_max, std::placeholders::_1));
                    m_map.emplace("ReduceMean", std::bind(op::reduce_mean, std::placeholders::_1));
                    m_map.emplace("ReduceMin", std::bind(op::reduce_min, std::placeholders::_1));
                    m_map.emplace("ReduceProd", std::bind(op::reduce_prod, std::placeholders::_1));
                    m_map.emplace("ReduceSum", std::bind(op::reduce_sum, std::placeholders::_1));
                    m_map.emplace("ReduceSumSquare",
                                  std::bind(op::reduce_sum_square, std::placeholders::_1));
                    m_map.emplace("Relu", std::bind(op::relu, std::placeholders::_1));
                    m_map.emplace("Reshape", std::bind(op::reshape, std::placeholders::_1));
                    m_map.emplace("Selu", std::bind(op::selu, std::placeholders::_1));
                    m_map.emplace("Shape", std::bind(op::shape, std::placeholders::_1));
                    m_map.emplace("Sigmoid", std::bind(op::sigmoid, std::placeholders::_1));
                    m_map.emplace("Slice", std::bind(op::slice, std::placeholders::_1));
                    m_map.emplace("Softmax", std::bind(op::softmax, std::placeholders::_1));
                    m_map.emplace("Softplus", std::bind(op::softplus, std::placeholders::_1));
                    m_map.emplace("Softsign", std::bind(op::softsign, std::placeholders::_1));
                    m_map.emplace("Split", std::bind(op::split, std::placeholders::_1));
                    m_map.emplace("Sqrt", std::bind(op::sqrt, std::placeholders::_1));
                    m_map.emplace("Squeeze", std::bind(op::squeeze, std::placeholders::_1));
                    m_map.emplace("Sub", std::bind(op::sub, std::placeholders::_1));
                    m_map.emplace("Sum", std::bind(op::sum, std::placeholders::_1));
                    m_map.emplace("Tanh", std::bind(op::tanh, std::placeholders::_1));
                    m_map.emplace("ThresholdedRelu",
                                  std::bind(op::thresholded_relu, std::placeholders::_1));
                    m_map.emplace("Transpose", std::bind(op::transpose, std::placeholders::_1));
                    m_map.emplace("Unsqueeze", std::bind(op::unsqueeze, std::placeholders::_1));
                    m_map.emplace("Xor", std::bind(op::logical_xor, std::placeholders::_1));
                }

                NodeVector operator()(const Node& node) const
                {
                    auto it = m_map.find(node.op_type());
                    if (it == m_map.end())
                    {
                        throw detail::error::unknown_operation{node.op_type()};
                    }

                    std::function<NodeVector(const Node&)> factory{it->second};
                    return factory(node);
                }
            };

        } // namespace detail

        namespace ops_bridge
        {
            NodeVector make_ng_nodes(const Node& node)
            {
                return detail::ops_bridge::make_ng_nodes(node);
            }

        } // namespace ops_bridge

    } // namespace onnx_import

} // namespace ngraph
