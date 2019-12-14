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

#include <limits>
#include <memory>

#include "clip.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/clamp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector clip(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);

                    const double max_value =
                        node.get_attribute_value<double>("max", std::numeric_limits<double>::max());

                    const double min_value = node.get_attribute_value<double>(
                        "min", std::numeric_limits<double>::lowest());

                    return {std::make_shared<ngraph::op::Clamp>(data, min_value, max_value)};
                }

            } // namespace set_1

            namespace set_11
            {
                NodeVector clip(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> data = inputs.at(0);
                    const element::Type data_type = data->get_element_type();
                    const Shape data_shape = data->get_shape();
                    std::shared_ptr<ngraph::Node> min;
                    std::shared_ptr<ngraph::Node> max;

                    // If second input is provided, assign to min input, otherwise set lowest
                    // numeric limit of double as min input.
                    if (inputs.size() > 1 && !inputs.at(1)->is_null())
                    {
                        min = inputs.at(1);
                    }
                    else
                    {
                        min = builder::make_constant(
                            data_type, data_shape, std::numeric_limits<double>::lowest());
                    }

                    // If third input is provided, assign to max input, otherwise set maximum
                    // numeric limit of double as max input.
                    if (inputs.size() == 3 && !inputs.at(2)->is_null())
                    {
                        max = inputs.at(2);
                    }
                    else
                    {
                        max = builder::make_constant(
                            data_type, data_shape, std::numeric_limits<double>::max());
                    }

                    auto max_of_min_and_data = std::make_shared<ngraph::op::Maximum>(
                        min,
                        data,
                        ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY));

                    return {std::make_shared<ngraph::op::Minimum>(
                        max,
                        max_of_min_and_data,
                        ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))};
                }

            } // namespace set_11

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
