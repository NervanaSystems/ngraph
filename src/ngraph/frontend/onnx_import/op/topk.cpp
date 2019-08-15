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

#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/type/element_type.hpp"
#include "topk.hpp"

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
                    std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};
                    std::int64_t k{node.get_attribute_value<std::int64_t>("k")};

                    auto num_dimensions = data->get_shape().size();

                    if (axis < 0)
                    {
                        axis += num_dimensions;
                    }

                    ASSERT_VALID_ARGUMENT(node, axis < num_dimensions)
                        << "`axis` parameter is out of range: " << axis;

                    std::shared_ptr<ngraph::Node> top_k =
                        std::make_shared<ngraph::op::TopK>(data, axis, element::i64, k);

                    std::shared_ptr<ngraph::Node> indices =
                        std::make_shared<ngraph::op::GetOutputElement>(top_k, 0);
                    std::shared_ptr<ngraph::Node> values =
                        std::make_shared<ngraph::op::GetOutputElement>(top_k, 1);

                    return {values, indices};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
