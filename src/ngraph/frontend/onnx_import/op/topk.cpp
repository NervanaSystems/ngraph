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

#include "ngraph/node.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/type/element_type.hpp"
#include "topk.hpp"
#include "utils/common.hpp"

static std::int64_t get_axis(const ngraph::onnx_import::Node& node)
{
    // Parse node attribute value for axis (adjust for negative value if needed).
    std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};

    auto data = node.get_ng_inputs().at(0);
    auto data_rank = data->get_shape().size();
    return ngraph::onnx_import::common::validate_axis(node, axis, data_rank);
}

static ngraph::NodeVector get_outputs(const std::shared_ptr<ngraph::Node>& top_k)
{
    std::shared_ptr<ngraph::Node> indices =
        std::make_shared<ngraph::op::GetOutputElement>(top_k, 0);
    std::shared_ptr<ngraph::Node> values = std::make_shared<ngraph::op::GetOutputElement>(top_k, 1);

    return {values, indices};
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
                    auto axis = get_axis(node);

                    std::shared_ptr<ngraph::Node> top_k =
                        std::make_shared<ngraph::op::TopK>(data, axis, element::i64, k);

                    return get_outputs(top_k);
                }
            }

            namespace set_10
            {
                NodeVector topk(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto k = node.get_ng_inputs().at(1);
                    auto axis = get_axis(node);

                    std::shared_ptr<ngraph::Node> top_k =
                        std::make_shared<ngraph::op::TopK>(data, k, axis, element::i64);

                    return get_outputs(top_k);
                }
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
