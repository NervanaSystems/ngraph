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

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "onehot.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector onehot(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto indices =
                        std::make_shared<default_opset::Convert>(inputs.at(0), element::i64);
                    auto depth = reshape::interpret_as_scalar(inputs.at(1));

                    auto values = inputs.at(2);
                    std::shared_ptr<ngraph::Node> off_value =
                        reshape::interpret_as_scalar(std::make_shared<ngraph::opset0::Slice>(
                            values, Coordinate{0}, Coordinate{1}));
                    std::shared_ptr<ngraph::Node> on_value =
                        reshape::interpret_as_scalar(std::make_shared<ngraph::opset0::Slice>(
                            values, Coordinate{1}, Coordinate{2}));

                    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);

                    return {std::make_shared<default_opset::OneHot>(
                        indices, depth, on_value, off_value, axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
