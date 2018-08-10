/*******************************************************************************
 * Copyright 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/node.hpp"
#include "ngraph/frontend/onnx_import/util/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/util/conv_pool.hpp"

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                std::shared_ptr<ngraph::op::Op>
                    make_ng_convolution(const std::shared_ptr<ngraph::Node>& data,
                                        const std::shared_ptr<ngraph::Node>& filters,
                                        const ngraph::Strides& strides,
                                        const ngraph::Strides& dilations,
                                        const ngraph::CoordinateDiff& padding_below,
                                        const ngraph::CoordinateDiff& padding_above,
                                        int groups);
            }

            /**
             * @brief Performs ONNX Conv operation.
             * 
             * @param node   The ONNX node object representing this operation.
             * 
             * @return The vector containing Ngraph nodes producing output of ONNX convolution 
             *         operation.
             */
            inline NodeVector conv(const Node& node)
            {
                const NodeVector& inputs = node.get_ng_inputs();
                auto data = inputs.at(0);
                auto filters = inputs.at(1);

                int groups{node.get_attribute_value<int>("group", 1)};
                if (groups < 0 || groups > data->get_shape().at(1) ||
                    groups > filters->get_shape().at(0))
                {
                    throw error::op::op_value_error("Conv",
                                                    node.get_name(),
                                                    "incorrect value of 'group' attribute: " +
                                                        std::to_string(groups));
                }

                auto strides = util::get_strides(node);
                auto dilations = util::get_dilations(node);
                auto paddings = util::get_pads(node);
                auto padding_below = paddings.first;
                auto padding_above = paddings.second;

                auto conv_node = detail::make_ng_convolution(
                    data, filters, strides, dilations, padding_below, padding_above, groups);

                // no bias param
                if (inputs.size() < 3)
                {
                    return NodeVector{conv_node};
                }

                auto bias = inputs.at(2);
                const Shape& new_shape = conv_node->get_shape();

                auto broadcasted_bias = std::make_shared<ngraph::op::Broadcast>(
                    bias, new_shape, util::get_broadcast_axes(new_shape, bias->get_shape(), 1));
                return NodeVector{std::make_shared<ngraph::op::Add>(conv_node, broadcasted_bias)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
