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
#include "op/add.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/div.hpp"
#include "op/flatten.hpp"
#include "op/gemm.hpp"
#include "op/matmul.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/mean.hpp"
#include "op/min.hpp"
#include "op/mul.hpp"
#include "op/pow.hpp"
#include "op/relu.hpp"
#include "op/reshape.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/unsqueeze.hpp"
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
                    m_map.emplace("Add", std::bind(op::add, std::placeholders::_1));
                    m_map.emplace("AveragePool",
                                  std::bind(op::average_pool, std::placeholders::_1));
                    m_map.emplace("BatchNormalization",
                                  std::bind(op::batch_norm, std::placeholders::_1));
                    m_map.emplace("Concat", std::bind(op::concat, std::placeholders::_1));
                    m_map.emplace("Constant", std::bind(op::constant, std::placeholders::_1));
                    m_map.emplace("Conv", std::bind(op::conv, std::placeholders::_1));
                    m_map.emplace("Div", std::bind(op::div, std::placeholders::_1));
                    m_map.emplace("Flatten", std::bind(op::flatten, std::placeholders::_1));
                    m_map.emplace("Gemm", std::bind(op::gemm, std::placeholders::_1));
                    m_map.emplace("MatMul", std::bind(op::matmul, std::placeholders::_1));
                    m_map.emplace("MaxPool", std::bind(op::max_pool, std::placeholders::_1));
                    m_map.emplace("Max", std::bind(op::max, std::placeholders::_1));
                    m_map.emplace("Mean", std::bind(op::mean, std::placeholders::_1));
                    m_map.emplace("Min", std::bind(op::min, std::placeholders::_1));
                    m_map.emplace("Mul", std::bind(op::mul, std::placeholders::_1));
                    m_map.emplace("Pow", std::bind(op::pow, std::placeholders::_1));
                    m_map.emplace("Relu", std::bind(op::relu, std::placeholders::_1));
                    m_map.emplace("Reshape", std::bind(op::reshape, std::placeholders::_1));
                    m_map.emplace("Softmax", std::bind(op::softmax, std::placeholders::_1));
                    m_map.emplace("Split", std::bind(op::split, std::placeholders::_1));
                    m_map.emplace("Sub", std::bind(op::sub, std::placeholders::_1));
                    m_map.emplace("Sum", std::bind(op::sum, std::placeholders::_1));
                    m_map.emplace("Unsqueeze", std::bind(op::unsqueeze, std::placeholders::_1));
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
