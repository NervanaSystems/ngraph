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

#include <cmath>
#include <cstddef>

#include "exceptions.hpp"
#include "instance_norm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector instance_norm(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    std::shared_ptr<ngraph::Node> scale{node.get_ng_inputs().at(1)};
                    std::shared_ptr<ngraph::Node> bias{node.get_ng_inputs().at(2)};
                    const float epsilon{node.get_attribute_value<float>("epsilon", 1e-5f)};

                    CHECK_VALID_NODE(node,
                                     (scale->get_shape().size() == 1 &&
                                      scale->get_shape()[0] == data->get_shape().at(1)),
                                     "Scale input must be one dimensional vector of number of "
                                     "input data channels size.");

                    CHECK_VALID_NODE(node,
                                     (bias->get_shape().size() == 1 &&
                                      bias->get_shape()[0] == data->get_shape().at(1)),
                                     "Bias input must be one dimensional vector of number of "
                                     "input data channels size.");

                    // all dimensions except spatial/feature
                    const AxisSet reduction_axes{
                        common::get_monotonic_range<std::size_t>(data->get_shape().size(), 2)};

                    const std::shared_ptr<ngraph::Node> eps_node =
                        std::make_shared<ngraph::op::Constant>(data->get_element_type(),
                                                               data->get_shape(),
                                                               std::vector<float>{epsilon});

                    scale = ngraph::op::legacy_style_broadcast_for_binary_operation(data, scale, 1)
                                .at(1);
                    bias = ngraph::op::legacy_style_broadcast_for_binary_operation(data, bias, 1)
                               .at(1);

                    std::shared_ptr<ngraph::Node> mean = builder::mean(data, reduction_axes);
                    mean = std::make_shared<ngraph::op::Broadcast>(
                        mean, data->get_shape(), reduction_axes);
                    std::shared_ptr<ngraph::Node> variance =
                        builder::variance(data, reduction_axes);
                    variance = std::make_shared<ngraph::op::Broadcast>(
                        variance, data->get_shape(), reduction_axes);

                    const auto sqrt = std::make_shared<ngraph::op::Sqrt>(variance + eps_node);

                    return {scale * (data - mean) / sqrt + bias};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
