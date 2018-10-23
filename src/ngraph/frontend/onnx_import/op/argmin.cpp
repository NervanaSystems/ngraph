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

#include "ngraph/op/argmin.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/reshape.hpp"

#include "core/attribute.hpp"
#include "core/node.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector argmin(const Node& node)
                {
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    auto keepdims = node.get_attribute_value<int64_t>("keepdims", 1);
                    auto input_node = node.get_ng_inputs().at(0);

                    auto op_node =
                        std::make_shared<ngraph::op::ArgMin>(input_node, axis, element::i64);

                    if (keepdims == 0)
                    {
                        return {op_node};
                    }

                    auto output_shape = input_node->get_shape();
                    output_shape.at(axis) = 1;

                    return {std::make_shared<ngraph::op::Reshape>(
                        op_node,
                        reshape::get_default_axis_vector(op_node->get_shape().size()),
                        Shape{output_shape})};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
