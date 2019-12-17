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

#include <cstdint>
#include <vector>

#include "default_opset.hpp"
#include "split.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector split(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    const auto axis_node =
                        default_opset::Constant::create(element::i64, Shape{}, {axis});

                    std::shared_ptr<ngraph::Node> split;
                    if (node.has_attribute("split"))
                    {
                        const auto splits =
                            node.get_attribute_value<std::vector<std::size_t>>("split");

                        const auto split_lengths = default_opset::Constant::create(
                            element::u64, Shape{splits.size()}, splits);

                        split = std::make_shared<default_opset::VariadicSplit>(
                            input, axis_node, split_lengths);
                    }
                    else
                    {
                        const auto outputs_number = node.get_output_names().size();
                        split = std::make_shared<default_opset::Split>(
                            input, axis_node, outputs_number);
                    }
                    return common::get_outputs(split);
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
