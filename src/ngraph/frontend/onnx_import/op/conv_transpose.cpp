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
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "conv_transpose.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector conv_transpose(const Node& node)
                {
                    const NodeVector& inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    auto filters = inputs.at(1);

                    const Shape& data_shape = data->get_shape();
                    const Shape& weights_shape = filters->get_shape();
                    int num_spatial_dims = data_shape.size() - 2;

                    auto strides = convpool::get_strides(node);
                    auto dilations = convpool::get_dilations(node);
                    auto paddings = convpool::get_pads(node);
                    ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    CoordinateDiff pads_begin = paddings.first;
                    CoordinateDiff pads_end = paddings.second;

                    std::vector<std::int64_t> output_shape{
                        node.get_attribute_value<std::vector<std::int64_t>>("output_shape", {})};

                    std::vector<std::int64_t> output_padding{
                        node.get_attribute_value<std::vector<std::int64_t>>(
                            "output_padding", std::vector<std::int64_t>(num_spatial_dims, 0))};

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    CHECK_VALID_NODE(node,
                                     ((groups >= 0) &&
                                      (groups <= static_cast<int64_t>(data->get_shape().at(1))) &&
                                      (groups <= static_cast<int64_t>(filters->get_shape().at(0)))),
                                     "incorrect value of 'group' attribute: ",
                                     groups);

                    std::size_t n_data_channels{data_shape.at(1)};
                    std::size_t n_filters_channels{weights_shape.at(0)};

                    CHECK_VALID_NODE(
                        node,
                        n_data_channels % groups == 0,
                        "provided group attribute value must be a multiple of data channels "
                        "count.");
                    CHECK_VALID_NODE(
                        node,
                        n_filters_channels % groups == 0,
                        "provided group attribute value must be a multiple of filter channels "
                        "count.");

                    // reshape filters to match desired shape:
                    // [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                    // from [C_INPUT x C_OUTPUT/groups x k1 x k2 x ... x kn]

                    Shape new_filters_shape{weights_shape};
                    new_filters_shape.at(0) /= groups;
                    new_filters_shape.insert(std::begin(new_filters_shape), groups);
                    filters = builder::opset1::reshape(filters, new_filters_shape);

                    std::shared_ptr<ngraph::Node> conv_node;
                    if (!output_shape.empty())
                    {
                        CHECK_VALID_NODE(
                            node,
                            output_shape.size() == num_spatial_dims,
                            "incorrect output_shape size. Got: ",
                            output_shape.size(),
                            ", but expected: ",
                            num_spatial_dims,
                            "output_shape attribute must be defined for all and only input data "
                            "spatial dimensions");

                        conv_node = std::make_shared<default_opset::GroupConvolutionBackpropData>(
                            data,
                            filters,
                            default_opset::Constant::create(
                                element::i64, Shape{output_shape.size()}, output_shape),
                            strides,
                            dilations,
                            auto_pad_type,
                            CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
                    }
                    else
                    {
                        conv_node = std::make_shared<default_opset::GroupConvolutionBackpropData>(
                            data,
                            filters,
                            strides,
                            pads_begin,
                            pads_end,
                            dilations,
                            auto_pad_type,
                            CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
                    }

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }

                    auto bias = inputs.at(2);
                    // Prepare bias shape [1, C, 1, 1]
                    Shape new_shape(conv_node->get_shape().size(), 1);
                    new_shape[1] = conv_node->get_shape()[1];

                    auto reshaped_bias = std::make_shared<default_opset::Reshape>(
                        bias,
                        default_opset::Constant::create(
                            element::i64, Shape{new_shape.size()}, new_shape),
                        true);

                    return {std::make_shared<default_opset::Add>(conv_node, reshaped_bias)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
