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

#include <functional>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/node.hpp"
#include "reduce.hpp"
#include "utils/reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector reduce_log_sum(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> sum_node{reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceSum,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                    return {std::make_shared<default_opset::Log>(sum_node)};
                }

                OutputVector reduce_log_sum_exp(const Node& node)
                {
                    auto exp_node =
                        std::make_shared<default_opset::Exp>(node.get_ng_inputs().at(0));
                    std::shared_ptr<ngraph::Node> sum_node{reduction::make_ng_reduction_op(
                        node,
                        exp_node,
                        std::make_shared<default_opset::ReduceSum,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                    return {std::make_shared<default_opset::Log>(sum_node)};
                }

                OutputVector reduce_l1(const Node& node)
                {
                    auto l1_norm_reduction = [](const Output<ngraph::Node>& node,
                                                const ngraph::AxisSet& axis_set) {
                        const auto axis_set_const = default_opset::Constant::create(
                            element::i64, {axis_set.size()}, axis_set.to_vector());
                        return ngraph::builder::opset1::l1_norm(node, axis_set_const, 0.f);
                    };

                    return {reduction::make_ng_reduction_op(
                        node, node.get_ng_inputs().at(0), l1_norm_reduction)};
                }

                OutputVector reduce_l2(const Node& node)
                {
                    auto l2_norm_reduction = [](const Output<ngraph::Node>& node,
                                                const ngraph::AxisSet& axis_set) {
                        const auto axis_set_const = default_opset::Constant::create(
                            element::i64, {axis_set.size()}, axis_set.to_vector());
                        return ngraph::builder::opset1::l2_norm(
                            node, axis_set_const, 0.f, ngraph::builder::BiasMode::ADD, false);
                    };
                    return {reduction::make_ng_reduction_op(
                        node, node.get_ng_inputs().at(0), l2_norm_reduction)};
                }

                OutputVector reduce_max(const Node& node)
                {
                    return {reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceMax,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }

                OutputVector reduce_mean(const Node& node)
                {
                    return {reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceMean,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }

                OutputVector reduce_min(const Node& node)
                {
                    return {reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceMin,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }

                OutputVector reduce_prod(const Node& node)
                {
                    return {reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceProd,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }

                OutputVector reduce_sum(const Node& node)
                {
                    return {reduction::make_ng_reduction_op(
                        node,
                        node.get_ng_inputs().at(0),
                        std::make_shared<default_opset::ReduceSum,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }

                OutputVector reduce_sum_square(const Node& node)
                {
                    auto input = Output<ngraph::Node>{node.get_ng_inputs().at(0)};
                    auto square_node = std::make_shared<default_opset::Multiply>(input, input);
                    return {reduction::make_ng_reduction_op(
                        node,
                        square_node,
                        std::make_shared<default_opset::ReduceSum,
                                         const Output<ngraph::Node>&,
                                         const Output<ngraph::Node>&,
                                         bool>)};
                }
            }
        }
    }
}
