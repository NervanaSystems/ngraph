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

#include "ngraph/op/fused/split.hpp"
#include "op/split.hpp"
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
                    const auto outputs_number = node.get_output_names().size();
                    const auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    std::size_t valid_axis =
                        common::validate_axis(node, axis, input->get_shape().size());

                    try
                    {
                        const auto length_parts =
                            node.get_attribute_value<std::vector<std::size_t>>("split");
                        const auto fused_split =
                            std::make_shared<ngraph::op::Split>(input, valid_axis, length_parts);

                        return fused_split->decompose_op();
                    }
                    catch (const error::node::UnknownAttribute&)
                    {
                        // an exception will be caught if the input node does not contain
                        // the 'split' attribute - this means we should split the input tensor
                        // into same-length parts equal to the number of node outputs
                        const auto fused_split =
                            std::make_shared<ngraph::op::Split>(input, valid_axis, outputs_number);

                        return fused_split->decompose_op();
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
