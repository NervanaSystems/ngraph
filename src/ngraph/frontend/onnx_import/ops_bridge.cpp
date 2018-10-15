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
#include <string>

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

                static bool is_op_type_supported(const std::string& op_type)
                {
                    return ops_bridge::get().is_op_type_supported_(op_type);
                }

            private:
                std::map<std::string, std::function<NodeVector(const Node&)>> m_map;

                static const ops_bridge& get()
                {
                    static ops_bridge instance;
                    return instance;
                }

#define REGISTER_OPERATOR(name_, version_, fn_)                                                    \
    m_map.emplace(name_, std::bind(op::set_##version_::fn_, std::placeholders::_1))

                ops_bridge()
                {
                    REGISTER_OPERATOR("Abs", 1, abs);
                    REGISTER_OPERATOR("Add", 1, add);
                    REGISTER_OPERATOR("And", 1, logical_and);
                    REGISTER_OPERATOR("AveragePool", 1, average_pool);
                    REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
                    REGISTER_OPERATOR("Cast", 1, cast);
                    REGISTER_OPERATOR("Ceil", 1, ceil);
                    REGISTER_OPERATOR("Clip", 1, clip);
                    REGISTER_OPERATOR("Concat", 1, concat);
                    REGISTER_OPERATOR("Constant", 1, constant);
                    REGISTER_OPERATOR("Conv", 1, conv);
                    REGISTER_OPERATOR("Div", 1, div);
                    REGISTER_OPERATOR("Dropout", 1, identity);
                    REGISTER_OPERATOR("Elu", 1, elu);
                    REGISTER_OPERATOR("Equal", 1, equal);
                    REGISTER_OPERATOR("Exp", 1, exp);
                    REGISTER_OPERATOR("Flatten", 1, flatten);
                    REGISTER_OPERATOR("Floor", 1, floor);
                    REGISTER_OPERATOR("Gemm", 1, gemm);
                    REGISTER_OPERATOR("Greater", 1, greater);
                    REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
                    REGISTER_OPERATOR("Identity", 1, identity);
                    REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
                    REGISTER_OPERATOR("Less", 1, less);
                    REGISTER_OPERATOR("Log", 1, log);
                    REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
                    REGISTER_OPERATOR("LRN", 1, lrn);
                    REGISTER_OPERATOR("MatMul", 1, matmul);
                    REGISTER_OPERATOR("MaxPool", 1, max_pool);
                    REGISTER_OPERATOR("Max", 1, max);
                    REGISTER_OPERATOR("Mean", 1, mean);
                    REGISTER_OPERATOR("Min", 1, min);
                    REGISTER_OPERATOR("Mul", 1, mul);
                    REGISTER_OPERATOR("Neg", 1, neg);
                    REGISTER_OPERATOR("Not", 1, logical_not);
                    REGISTER_OPERATOR("Or", 1, logical_or);
                    REGISTER_OPERATOR("Pow", 1, pow);
                    REGISTER_OPERATOR("PRelu", 1, prelu);
                    REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
                    REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
                    REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
                    REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
                    REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
                    REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
                    REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
                    REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
                    REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
                    REGISTER_OPERATOR("ReduceSum", 1, reduce_sum);
                    REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
                    REGISTER_OPERATOR("Relu", 1, relu);
                    REGISTER_OPERATOR("Reshape", 1, reshape);
                    REGISTER_OPERATOR("Selu", 1, selu);
                    REGISTER_OPERATOR("Shape", 1, shape);
                    REGISTER_OPERATOR("Sigmoid", 1, sigmoid);
                    REGISTER_OPERATOR("Slice", 1, slice);
                    REGISTER_OPERATOR("Softmax", 1, softmax);
                    REGISTER_OPERATOR("Softplus", 1, softplus);
                    REGISTER_OPERATOR("Softsign", 1, softsign);
                    REGISTER_OPERATOR("Split", 1, split);
                    REGISTER_OPERATOR("Sqrt", 1, sqrt);
                    REGISTER_OPERATOR("Squeeze", 1, squeeze);
                    REGISTER_OPERATOR("Sub", 1, sub);
                    REGISTER_OPERATOR("Sum", 1, sum);
                    REGISTER_OPERATOR("Tanh", 1, tanh);
                    REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
                    REGISTER_OPERATOR("Transpose", 1, transpose);
                    REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
                    REGISTER_OPERATOR("Xor", 1, logical_xor);
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

                bool is_op_type_supported_(const std::string& op_type) const
                {
                    auto it = m_map.find(op_type);
                    return !(it == m_map.end());
                }
            };

        } // namespace detail

        namespace ops_bridge
        {
            NodeVector make_ng_nodes(const Node& node)
            {
                return detail::ops_bridge::make_ng_nodes(node);
            }

            bool is_op_type_supported(const std::string& op_type)
            {
                return detail::ops_bridge::is_op_type_supported(op_type);
            }

        } // namespace ops_bridge

    } // namespace onnx_import

} // namespace ngraph
