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
#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "topk.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace
{
    /// \return Parse node attribute value for axis and adjust for negative value if needed.
    std::int64_t get_axis(const ngraph::onnx_import::Node& node)
    {
        std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};

        auto data = node.get_ng_inputs().at(0);
        auto data_rank = data->get_shape().size();
        return ngraph::onnx_import::common::validate_axis(node, axis, data_rank);
    }

    /// \return Return the second input to the TopK node reshaped to a scalar.
    std::shared_ptr<ngraph::Node> get_k(const ngraph::onnx_import::Node& node)
    {
        auto k_node = node.get_ng_inputs().at(1);
        NGRAPH_CHECK(shape_size(k_node->get_shape()) == 1,
                     "ONNX TopK operator: 'K' parameter must contain a single positive value.",
                     node);

        return ngraph::onnx_import::reshape::interpret_as_scalar(k_node);
    }

    /// \return Return the outputs of the TopK node.
    ngraph::NodeVector get_outputs(const std::shared_ptr<ngraph::Node>& node)
    {
        std::shared_ptr<ngraph::Node> values =
            std::make_shared<ngraph::opset0::GetOutputElement>(node, 0);
        std::shared_ptr<ngraph::Node> indices =
            std::make_shared<ngraph::opset0::GetOutputElement>(node, 1);

        return {values, indices};
    }
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector topk(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    std::int64_t k{node.get_attribute_value<std::int64_t>("k")};
                    auto k_node = default_opset::Constant::create(element::i64, Shape{}, {k});
                    auto axis = get_axis(node);

                    std::shared_ptr<ngraph::Node> top_k = std::make_shared<default_opset::TopK>(
                        data,
                        k_node,
                        axis,
                        default_opset::TopK::Mode::MAX,
                        default_opset::TopK::SortType::SORT_VALUES,
                        element::i64);

                    return get_outputs(top_k);
                }
            }

            namespace set_10
            {
                NodeVector topk(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto k = get_k(node);
                    auto axis = get_axis(node);

                    std::shared_ptr<ngraph::Node> top_k = std::make_shared<default_opset::TopK>(
                        data,
                        k,
                        axis,
                        default_opset::TopK::Mode::MAX,
                        default_opset::TopK::SortType::SORT_VALUES,
                        element::i64);

                    return get_outputs(top_k);
                }
            }

            namespace set_11
            {
                NodeVector topk(const Node& node)
                {
                    // Process inputs
                    auto data = node.get_ng_inputs().at(0);
                    auto k = get_k(node);

                    // Process attributes
                    const auto axis = get_axis(node);
                    const auto largest = node.get_attribute_value<std::int64_t>("largest", 1);
                    const auto sorted = node.get_attribute_value<std::int64_t>("sorted", 1);

                    // Map attribute values to nGraph enums
                    const auto sort_type = sorted ? default_opset::TopK::SortType::SORT_VALUES
                                                  : default_opset::TopK::SortType::NONE;

                    const auto compute_max = static_cast<bool>(largest);
                    const auto mode = compute_max ? default_opset::TopK::Mode::MAX
                                                  : default_opset::TopK::Mode::MIN;

                    std::shared_ptr<ngraph::Node> top_k = std::make_shared<default_opset::TopK>(
                        data, k, axis, mode, sort_type, element::i64);

                    return get_outputs(top_k);
                }
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
