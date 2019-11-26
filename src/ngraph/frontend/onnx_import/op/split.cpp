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

#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "op/split.hpp"
#include "utils/common.hpp"

/// \return Return the outputs of the node.
static ngraph::NodeVector get_outputs(const std::shared_ptr<ngraph::Node>& node,
                                      int64_t outputs_numer)
{
    ngraph::NodeVector outputs(outputs_numer);
    for (int i = 0; i < outputs_numer; ++i)
    {
        outputs[i] = std::make_shared<ngraph::op::GetOutputElement>(node, i);
    }
    return outputs;
}

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
                        ngraph::op::Constant::create(element::i64, Shape{}, {axis});

                    std::shared_ptr<ngraph::op::Split> fused_split;
                    try
                    {
                        const auto length_parts =
                            node.get_attribute_value<std::vector<std::size_t>>("split");
                        fused_split =
                            std::make_shared<ngraph::op::Split>(input, axis_node, length_parts);
                    }
                    catch (const error::node::UnknownAttribute&)
                    {
                        // an exception will be caught if the input node does not contain
                        // the 'split' attribute - this means we should split the input tensor
                        // into same-length parts equal to the number of node outputs
                        const auto outputs_number = node.get_output_names().size();
                        fused_split =
                            std::make_shared<ngraph::op::Split>(input, axis_node, outputs_number);
                    }
                    return get_outputs(fused_split, fused_split->get_splits().size());
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
