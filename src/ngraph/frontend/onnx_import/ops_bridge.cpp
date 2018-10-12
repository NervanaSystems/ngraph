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

#define REGISTER_OPERATOR(name_, fn_)                                                              \
    m_map.emplace(name_, std::bind(op::fn_, std::placeholders::_1))

                ops_bridge()
                {
                    REGISTER_OPERATOR("Abs", abs);
                    REGISTER_OPERATOR("Add", add);
                    REGISTER_OPERATOR("And", logical_and);
                    REGISTER_OPERATOR("AveragePool", average_pool);
                    REGISTER_OPERATOR("BatchNormalization", batch_norm);
                    REGISTER_OPERATOR("Cast", cast);
                    REGISTER_OPERATOR("Ceil", ceil);
                    REGISTER_OPERATOR("Clip", clip);
                    REGISTER_OPERATOR("Concat", concat);
                    REGISTER_OPERATOR("Constant", constant);
                    REGISTER_OPERATOR("Conv", conv);
                    REGISTER_OPERATOR("Div", div);
                    REGISTER_OPERATOR("Dropout", identity);
                    REGISTER_OPERATOR("Elu", elu);
                    REGISTER_OPERATOR("Equal", equal);
                    REGISTER_OPERATOR("Exp", exp);
                    REGISTER_OPERATOR("Flatten", flatten);
                    REGISTER_OPERATOR("Floor", floor);
                    REGISTER_OPERATOR("Gemm", gemm);
                    REGISTER_OPERATOR("Greater", greater);
                    REGISTER_OPERATOR("HardSigmoid", hard_sigmoid);
                    REGISTER_OPERATOR("Identity", identity);
                    REGISTER_OPERATOR("LeakyRelu", leaky_relu);
                    REGISTER_OPERATOR("Less", less);
                    REGISTER_OPERATOR("Log", log);
                    REGISTER_OPERATOR("LogSoftmax", log_softmax);
                    REGISTER_OPERATOR("LRN", lrn);
                    REGISTER_OPERATOR("MatMul", matmul);
                    REGISTER_OPERATOR("MaxPool", max_pool);
                    REGISTER_OPERATOR("Max", max);
                    REGISTER_OPERATOR("Mean", mean);
                    REGISTER_OPERATOR("Min", min);
                    REGISTER_OPERATOR("Mul", mul);
                    REGISTER_OPERATOR("Neg", neg);
                    REGISTER_OPERATOR("Not", logical_not);
                    REGISTER_OPERATOR("Or", logical_or);
                    REGISTER_OPERATOR("Pow", pow);
                    REGISTER_OPERATOR("PRelu", prelu);
                    REGISTER_OPERATOR("Reciprocal", reciprocal);
                    REGISTER_OPERATOR("ReduceLogSum", reduce_log_sum);
                    REGISTER_OPERATOR("ReduceLogSumExp", reduce_log_sum_exp);
                    REGISTER_OPERATOR("ReduceL1", reduce_l1);
                    REGISTER_OPERATOR("ReduceL2", reduce_l2);
                    REGISTER_OPERATOR("ReduceMax", reduce_max);
                    REGISTER_OPERATOR("ReduceMean", reduce_mean);
                    REGISTER_OPERATOR("ReduceMin", reduce_min);
                    REGISTER_OPERATOR("ReduceProd", reduce_prod);
                    REGISTER_OPERATOR("ReduceSum", reduce_sum);
                    REGISTER_OPERATOR("ReduceSumSquare", reduce_sum_square);
                    REGISTER_OPERATOR("Relu", relu);
                    REGISTER_OPERATOR("Reshape", reshape);
                    REGISTER_OPERATOR("Selu", selu);
                    REGISTER_OPERATOR("Shape", shape);
                    REGISTER_OPERATOR("Sigmoid", sigmoid);
                    REGISTER_OPERATOR("Slice", slice);
                    REGISTER_OPERATOR("Softmax", softmax);
                    REGISTER_OPERATOR("Softplus", softplus);
                    REGISTER_OPERATOR("Softsign", softsign);
                    REGISTER_OPERATOR("Split", split);
                    REGISTER_OPERATOR("Sqrt", sqrt);
                    REGISTER_OPERATOR("Squeeze", squeeze);
                    REGISTER_OPERATOR("Sub", sub);
                    REGISTER_OPERATOR("Sum", sum);
                    REGISTER_OPERATOR("Tanh", tanh);
                    REGISTER_OPERATOR("ThresholdedRelu", thresholded_relu);
                    REGISTER_OPERATOR("Transpose", transpose);
                    REGISTER_OPERATOR("Unsqueeze", unsqueeze);
                    REGISTER_OPERATOR("Xor", logical_xor);
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
