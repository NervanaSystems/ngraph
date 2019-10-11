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

#include "mean_variance_normalization.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector mean_variance_normalization(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    bool across_channels =
                        node.get_attribute_value<std::int64_t>("across_channels", 0);
                    bool normalize_variance =
                        node.get_attribute_value<std::int64_t>("normalize_variance", 1);

                    return {std::make_shared<ngraph::op::MVN>(
                        data, across_channels, normalize_variance)};
                }

            } // namespace set_1

            namespace set_9
            {
                NodeVector mean_variance_normalization(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {0, 2, 3});
                    std::vector<std::size_t> valid_axes =
                        common::validate_axes(node, axes, data->get_shape().size());

                    return {std::make_shared<ngraph::op::MVN>(data, AxisSet(valid_axes))};
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
