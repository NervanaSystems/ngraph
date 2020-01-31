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

#include <memory>

#include "default_opset.hpp"
#include "core/null_node.hpp"
#include "max_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector max_pool(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto kernel_shape =
                        node.get_attribute_value<std::vector<std::size_t>>("kernel_shape");
                    const auto strides = convpool::get_strides(node);
                    const auto auto_pad = convpool::get_auto_pad(node);
                    CoordinateDiff padding_below, padding_above;
                    std::tie(padding_below, padding_above) = convpool::get_pads(node);

                    auto max_pool = NodeVector{std::make_shared<default_opset::MaxPool>(
                        data,
                        strides,
                        Shape{std::begin(padding_below), std::end(padding_below)},
                        Shape{std::begin(padding_above), std::end(padding_above)},
                        kernel_shape,
                        ngraph::op::RoundingType::FLOOR,
                        auto_pad)};

                    max_pool.emplace_back(std::make_shared<NullNode>()); // Indices (optional)
                    return max_pool;
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
