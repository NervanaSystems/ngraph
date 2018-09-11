//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <limits>
#include <memory>

#include "ngraph/node.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

#include "clip.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector clip(const Node& node)
            {
                auto data = node.get_ng_inputs().at(0);

                double max_value =
                    node.get_attribute_value<double>("max", std::numeric_limits<double>::max());
                double min_value =
                    node.get_attribute_value<double>("min", std::numeric_limits<double>::lowest());

                std::shared_ptr<ngraph::Node> max_value_node =
                    std::make_shared<ngraph::op::Constant>(
                        data->get_element_type(), ngraph::Shape{}, std::vector<double>{max_value});
                max_value_node = make_broadcast_node(max_value_node, data->get_shape());

                std::shared_ptr<ngraph::Node> min_value_node =
                    std::make_shared<ngraph::op::Constant>(
                        data->get_element_type(), ngraph::Shape{}, std::vector<double>{min_value});
                min_value_node = make_broadcast_node(min_value_node, data->get_shape());

                return {std::make_shared<ngraph::op::Minimum>(
                    max_value_node, std::make_shared<ngraph::op::Maximum>(data, min_value_node))};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
