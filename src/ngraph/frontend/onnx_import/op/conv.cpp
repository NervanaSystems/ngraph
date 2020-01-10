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
#include "ngraph/opsets/opset0.hpp"
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
                                            int groups,
                                            const ngraph::op::PadType& auto_pad)
                    {
                        if (groups > 1)
                        {
                            auto filters_shape = filters->get_shape();
                            filters_shape.at(0) = filters_shape.at(0) / groups;
                            filters_shape.insert(filters_shape.begin(), groups);

                            auto reshaped_filters =
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

                } // namespace

                NodeVector conv(const Node& node)
                {
                    const NodeVector& inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    auto filters = inputs.at(1);

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    ASSERT_VALID_ARGUMENT(
                        node,
                        ((groups >= 0) &&
                         (groups <= static_cast<int64_t>(data->get_shape().at(1))) &&
                         (groups <= static_cast<int64_t>(filters->get_shape().at(0)))))
                        << "incorrect value of 'group' attribute: " << groups;

                    std::size_t n_data_channels{data->get_shape().at(1)};
                    std::size_t n_filters_channels{filters->get_shape().at(0)};

                    ASSERT_VALID_ARGUMENT(node, n_data_channels % groups == 0)
                        << "provided group attribute value must be a multiple of data channels "
                           "count.";
                    ASSERT_VALID_ARGUMENT(node, n_filters_channels % groups == 0)
                        << "provided group attribute value must be a multiple of filter channels "
                           "count.";

                    auto strides = convpool::get_strides(node);
                    auto dilations = convpool::get_dilations(node);
                    auto paddings = convpool::get_pads(node);
                    ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    const auto& padding_below = paddings.first;
                    const auto& padding_above = paddings.second;

                    auto conv_node = make_ng_convolution(data,
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

                    auto bias = inputs.at(2);
                    const Shape& new_shape = conv_node->get_shape();

                    auto broadcasted_bias = std::make_shared<default_opset::Broadcast>(
                        bias,
                        default_opset::Constant::create(
                            element::i64, Shape{new_shape.size()}, new_shape),
                        default_opset::Constant::create(element::i64, Shape{1}, {1}));
                    return {std::make_shared<default_opset::Add>(conv_node, broadcasted_bias)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
