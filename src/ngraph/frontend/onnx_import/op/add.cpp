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

#include "add.hpp"
#include "default_opset.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector add(const Node& node)
                {
                    auto lhs_node = node.get_ng_inputs().at(0);
                    auto rhs_node = node.get_ng_inputs().at(1);
                    auto lhs_shape = lhs_node->get_shape();
                    auto rhs_shape = rhs_node->get_shape();
                    auto lhs_rank = lhs_shape.size();
                    auto rhs_rank = rhs_shape.size();
                    auto axis =
                        node.get_attribute_value<std::int64_t>("axis", lhs_rank - rhs_rank);
                    // Unidirectional broadcast right node to left shape.
                    rhs_node = std::make_shared<default_opset::Broadcast>(
                        rhs_node,
                        default_opset::Constant::create(
                            ngraph::element::i64, Shape{lhs_rank}, lhs_shape),
                        ngraph::op::opset1::get_axes_mapping_output(lhs_shape, rhs_shape, axis));

                    return {std::make_shared<default_opset::Add>(
                        lhs_node, rhs_node, ngraph::op::AutoBroadcastSpec::NONE)};
                }

            } // namespace set_1

            namespace set_7
            {
                NodeVector add(const Node& node)
                {
                    return {std::make_shared<default_opset::Add>(node.get_ng_inputs().at(0),
                                                                 node.get_ng_inputs().at(1))};
                }

            } // namespace set_7

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
