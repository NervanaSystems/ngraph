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

#include <cstddef>
#include <memory>
#include <vector>

#include "conv.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
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
                                            int64_t groups,
                                            const ngraph::op::PadType& auto_pad)
                    {
                        if (groups > 1)
                        {
                            auto filters_shape = filters->get_shape();
                            filters_shape.at(0) = filters_shape.at(0) / groups;
                            filters_shape.insert(filters_shape.begin(), groups);

                            const auto reshaped_filters =
                                ngraph::builder::opset1::reshape(filters, filters_shape);

                            return std::make_shared<default_opset::GroupConvolution>(
                                data,
                                reshaped_filters,
                                strides,
                                padding_below,
                                padding_above,
                                dilations,
                                auto_pad);
                        }
                        else
                        {
                            return std::make_shared<default_opset::Convolution>(data,
                                                                                filters,
                                                                                strides,
                                                                                padding_below,
                                                                                padding_above,
                                                                                dilations,
                                                                                auto_pad);
                        }
                    }

                    void validate_conv_attributes(const PartialShape& data_ps,
                                                  const PartialShape& filters_ps,
                                                  const int64_t groups)
                    {
                        NGRAPH_CHECK(
                            groups >= 1,
                            "The 'groups' attribute of a Conv node must be greater or equal "
                            "to 1. Got: ",
                            groups);

                        NGRAPH_CHECK(data_ps.rank().is_static(),
                                     "The input data tensor's rank has to be known (static)");

                        NGRAPH_CHECK(filters_ps.rank().is_static(),
                                     "The filters(weights) tensor's rank has to be known (static)");

                        NGRAPH_CHECK(data_ps[1].is_static(),
                                     "The input data channel dimension has to be known (static)");

                        NGRAPH_CHECK(filters_ps[0].is_static(),
                                     "The number of feature maps has to be known (static)");

                        const auto n_data_channels = static_cast<size_t>(data_ps[1]);
                        const auto n_feature_maps = static_cast<size_t>(filters_ps[0]);
                        NGRAPH_CHECK(groups <= n_data_channels && groups <= n_feature_maps,
                                     "Incorrect value of 'group' attribute: ",
                                     groups);

                        NGRAPH_CHECK(n_data_channels % groups == 0,
                                     "Provided group attribute value must be a multiple of data "
                                     "channels count.");

                        NGRAPH_CHECK(n_feature_maps % groups == 0,
                                     "Provided group attribute value must be a multiple of filter "
                                     "channels count.");
                    }

                    std::shared_ptr<ngraph::Node>
                        add_bias(const std::shared_ptr<ngraph::Node>& ng_conv,
                                 const std::shared_ptr<ngraph::Node>& bias)
                    {
                        std::shared_ptr<ngraph::Node> broadcasted_bias;

                        if (ng_conv->is_dynamic())
                        {
                            const auto shape_of_conv =
                                std::make_shared<default_opset::ShapeOf>(ng_conv);

                            const auto rank_of_conv =
                                static_cast<size_t>(ng_conv->get_output_partial_shape(0).rank());

                            // reshape the bias node to {1, M, 1, 1, ..., 1} matching the rank of
                            // ng_conv
                            // this is required so that the following Broadcast can automatically
                            // handle broadcasting the bias the numpy way
                            std::vector<size_t> reshape_pattern_values(rank_of_conv, 1U);
                            reshape_pattern_values[1] = bias->get_shape().front();
                            const auto reshape_pattern = default_opset::Constant::create(
                                element::u64,
                                Shape{reshape_pattern_values.size()},
                                reshape_pattern_values);

                            const auto reshaped_bias = std::make_shared<default_opset::Reshape>(
                                bias, reshape_pattern, true);

                            broadcasted_bias = std::make_shared<default_opset::Broadcast>(
                                reshaped_bias,
                                shape_of_conv,
                                // the constant below will be ignored
                                default_opset::Constant::create(element::i64, Shape{}, {1}),
                                ngraph::op::AutoBroadcastSpec(
                                    ngraph::op::AutoBroadcastType::NUMPY));
                        }
                        else
                        {
                            const Shape& conv_shape = ng_conv->get_shape();
                            const auto target_shape = default_opset::Constant::create(
                                element::i64, Shape{conv_shape.size()}, conv_shape);

                            broadcasted_bias = std::make_shared<default_opset::Broadcast>(
                                bias,
                                target_shape,
                                default_opset::Constant::create(element::i64, Shape{1}, {1}));
                        }

                        return {std::make_shared<default_opset::Add>(ng_conv, broadcasted_bias)};
                    }
                } // namespace

                NodeVector conv(const Node& node)
                {
                    // in the current implementation we assume that the data input rank is static
                    // and only the 'batch' dimension can be dynamic
                    const NodeVector& inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    const auto filters = inputs.at(1);
                    const auto groups = node.get_attribute_value<int64_t>("group", 1);

                    validate_conv_attributes(data->get_output_partial_shape(0),
                                             filters->get_output_partial_shape(0),
                                             groups);

                    const auto strides = convpool::get_strides(node);
                    const auto dilations = convpool::get_dilations(node);
                    const auto paddings = convpool::get_pads(node);
                    const ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    const auto& padding_below = paddings.first;
                    const auto& padding_above = paddings.second;

                    const auto conv_node = make_ng_convolution(data,
                                                               filters,
                                                               strides,
                                                               dilations,
                                                               padding_below,
                                                               padding_above,
                                                               groups,
                                                               auto_pad_type);

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }
                    else
                    {
                        const auto bias = inputs.at(2);
                        const auto bias_ps = bias->get_output_partial_shape(0);

                        NGRAPH_CHECK(bias_ps.is_static() && is_vector(bias_ps.to_shape()),
                                     "The bias input needs to be a static 1D vector");

                        return {add_bias(conv_node, bias)};
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
