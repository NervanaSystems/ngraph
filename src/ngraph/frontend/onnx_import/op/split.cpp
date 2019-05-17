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

#include "exceptions.hpp"
#include "op/split.hpp"
#include "ngraph/op/fused/split.hpp"

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
                    const std::shared_ptr<ngraph::Node> input = node.get_ng_inputs().at(0);
                    const Shape input_shape = input->get_shape();
                    const std::size_t count_outputs{node.get_output_names().size()};
                    const int64_t axis{node.get_attribute_value<int64_t>("axis", 0)};

                    try
                    {
                        const auto length_parts = node.get_attribute_value<std::vector<std::size_t>>("split");
                        const auto fused_split = std::make_shared<ngraph::op::Split>(input, axis, length_parts);
                        return fused_split->decompose_op();
                    }
                    catch (const std::exception&)
                    {
                        const auto fused_split = std::make_shared<ngraph::op::Split>(input, axis, count_outputs);
                        return fused_split->decompose_op();

                    }
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
