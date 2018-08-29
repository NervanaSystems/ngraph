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

#include <cstddef>
#include <memory>
#include <vector>

#include "op/conv.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/slice.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/op/conv.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::shared_ptr<ngraph::op::Op>
                    make_ng_convolution(const std::shared_ptr<ngraph::Node>& data,
                                        const std::shared_ptr<ngraph::Node>& filters,
                                        const ngraph::Strides& strides,
                                        const ngraph::Strides& dilations,
                                        const ngraph::CoordinateDiff& padding_below,
                                        const ngraph::CoordinateDiff& padding_above,
                                        int groups)
                {
                    if (groups > 1)
                    {
                        // Split one convolution op to N ops where N is the number of groups
                        // and concat results after computation.
                        // reference: https://github.com/NervanaSystems/ngraph-mxnet/blob/fdd692/src/ngraph/ngraph_emitter.cc#L822-L856
                        std::size_t n_data_channels{data->get_shape().at(1)};
                        std::size_t n_filters_channels{filters->get_shape().at(0)};
                        // TODO: ensure n_data_channels % groups = 0
                        std::size_t data_group_size{n_data_channels / groups};
                        std::size_t filters_group_size{n_filters_channels / groups};
                        NodeVector convolution_nodes;

                        // initial bounds for splice
                        std::vector<std::size_t> data_lower_bounds(data->get_shape().size());
                        std::vector<std::size_t> data_upper_bounds{data->get_shape()};
                        std::vector<std::size_t> filters_lower_bounds(filters->get_shape().size());
                        std::vector<std::size_t> filters_upper_bounds{filters->get_shape()};

                        for (std::size_t group{0}; group < groups; ++group)
                        {
                            // slice data
                            data_lower_bounds[1] = group * data_group_size;
                            data_upper_bounds[1] = (group + 1) * data_group_size;
                            auto sliced_data = std::make_shared<ngraph::op::Slice>(
                                data, data_lower_bounds, data_upper_bounds);
                            // slice filters
                            filters_lower_bounds[0] = group * filters_group_size;
                            filters_upper_bounds[0] = (group + 1) * filters_group_size;
                            auto sliced_filters = std::make_shared<ngraph::op::Slice>(
                                filters, filters_lower_bounds, filters_upper_bounds);

                            convolution_nodes.push_back(
                                std::make_shared<ngraph::op::Convolution>(sliced_data,
                                                                          sliced_filters,
                                                                          strides,
                                                                          dilations,
                                                                          padding_below,
                                                                          padding_above));
                        }
                        std::size_t concatenation_axis = 1;
                        return std::make_shared<ngraph::op::Concat>(convolution_nodes,
                                                                    concatenation_axis);
                    }
                    else
                    {
                        return std::make_shared<ngraph::op::Convolution>(
                            data, filters, strides, dilations, padding_below, padding_above);
                    }
                }

            } // namespace

            NodeVector conv(const Node& node)
            {
                const NodeVector& inputs = node.get_ng_inputs();
                auto data = inputs.at(0);
                auto filters = inputs.at(1);

                int groups{node.get_attribute_value<int>("group", 1)};

                // TODO: update to ASSERTION CHECK
                if (groups < 0 || groups > data->get_shape().at(1) ||
                    groups > filters->get_shape().at(0))
                {
                    throw error::parameter::Value{"Conv",
                                                  node.get_name(),
                                                  "incorrect value of 'group' attribute: " +
                                                      std::to_string(groups)};
                }

                auto strides = convpool::get_strides(node);
                auto dilations = convpool::get_dilations(node);
                auto paddings = convpool::get_pads(node);
                const auto& padding_below = paddings.first;
                const auto& padding_above = paddings.second;

                auto conv_node = make_ng_convolution(
                    data, filters, strides, dilations, padding_below, padding_above, groups);

                // no bias param
                if (inputs.size() < 3)
                {
                    return {conv_node};
                }

                auto bias = inputs.at(2);
                const Shape& new_shape = conv_node->get_shape();

                auto broadcasted_bias = std::make_shared<ngraph::op::Broadcast>(
                    bias, new_shape, calculate_broadcast_axes(new_shape, bias->get_shape(), 1));
                return {std::make_shared<ngraph::op::Add>(conv_node, broadcasted_bias)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
