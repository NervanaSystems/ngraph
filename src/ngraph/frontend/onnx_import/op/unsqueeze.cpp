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
#include "ngraph/shape.hpp"
#include "unsqueeze.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector unsqueeze(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});
                    const auto expanded_rank = data->get_shape().size() + axes.size();
                    std::vector<std::size_t> valid_axes =
                        common::validate_axes(node, axes, expanded_rank);
                    auto axes_node = std::make_shared<default_opset::Constant>(
                        element::i64, Shape{valid_axes.size()}, valid_axes);
                    return {std::make_shared<default_opset::Unsqueeze>(data, axes_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
