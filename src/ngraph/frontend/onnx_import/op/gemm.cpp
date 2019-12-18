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

#include <memory>

#include "default_opset.hpp"
#include "gemm.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/matmul.hpp"
#include "ngraph/op/multiply.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector gemm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> input_a = inputs.at(0);
                    std::shared_ptr<ngraph::Node> input_b = inputs.at(1);
                    std::shared_ptr<ngraph::Node> input_c;

                    if (inputs.size() == 3)
                    {
                        input_c = inputs.at(2);
                    }
                    else
                    {
                        input_c = default_opset::Constant::create(
                            input_b->get_element_type(), ngraph::Shape{}, {0});
                    }

                    const auto alpha = node.get_attribute_value<float>("alpha", 1);
                    const auto beta = node.get_attribute_value<float>("beta", 1);

                    const auto alpha_node = default_opset::Constant::create(
                        element::Type_t::f32, Shape{}, std::vector<float>{alpha});
                    const auto beta_node = default_opset::Constant::create(
                        element::Type_t::f32, Shape{}, std::vector<float>{beta});

                    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    if (trans_a)
                    {
                        input_a = ngraph::builder::transpose(input_a);
                    }

                    if (trans_b)
                    {
                        input_b = ngraph::builder::transpose(input_b);
                    }

                    input_a = ngraph::builder::flatten(input_a, 1);
                    input_b = ngraph::builder::flatten(input_b, 1);

                    auto matmul_node = std::make_shared<ngraph::op::MatMul>(input_a, input_b);

                    auto alpha_times_product =
                        std::make_shared<default_opset::Multiply>(alpha_node, matmul_node);
                    auto beta_times_input_c =
                        std::make_shared<default_opset::Multiply>(beta_node, input_c);

                    return NodeVector{std::make_shared<default_opset::Add>(alpha_times_product,
                                                                           beta_times_input_c)};
                }

            } // namespace set_1

            namespace set_6
            {
                NodeVector gemm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> input_a = inputs.at(0);
                    std::shared_ptr<ngraph::Node> input_b = inputs.at(1);
                    std::shared_ptr<ngraph::Node> input_c;

                    if (inputs.size() == 3)
                    {
                        input_c = inputs.at(2);
                    }
                    else
                    {
                        input_c = default_opset::Constant::create(
                            input_b->get_element_type(), ngraph::Shape{}, {0});
                    }

                    const auto alpha = node.get_attribute_value<float>("alpha", 1);
                    const auto beta = node.get_attribute_value<float>("beta", 1);

                    const auto alpha_node = default_opset::Constant::create(
                        element::Type_t::f32, Shape{}, std::vector<float>{alpha});
                    const auto beta_node = default_opset::Constant::create(
                        element::Type_t::f32, Shape{}, std::vector<float>{beta});

                    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    auto matmul_node =
                        std::make_shared<default_opset::MatMul>(input_a, input_b, trans_a, trans_b);

                    auto alpha_times_product =
                        std::make_shared<default_opset::Multiply>(alpha_node, matmul_node);
                    auto beta_times_input_c =
                        std::make_shared<default_opset::Multiply>(beta_node, input_c);

                    return NodeVector{std::make_shared<default_opset::Add>(alpha_times_product,
                                                                           beta_times_input_c)};
                }

            } // namespace set_6

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
