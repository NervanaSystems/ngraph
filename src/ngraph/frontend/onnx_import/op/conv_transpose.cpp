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
#include "exceptions.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/fused/group_conv_transpose.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/opsets/opset0.hpp"
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
                    CoordinateDiff padding_below = paddings.first;
                    CoordinateDiff padding_above = paddings.second;

                    std::vector<std::int64_t> output_shape{
                        node.get_attribute_value<std::vector<std::int64_t>>("output_shape", {})};

                    std::vector<std::int64_t> output_padding{
                        node.get_attribute_value<std::vector<std::int64_t>>(
                            "output_padding", std::vector<std::int64_t>(num_spatial_dims, 0))};

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    ASSERT_VALID_ARGUMENT(
                        node,
                        ((groups >= 0) &&
                         (groups <= static_cast<int64_t>(data->get_shape().at(1))) &&
                         (groups <= static_cast<int64_t>(filters->get_shape().at(0)))))
                        << "incorrect value of 'group' attribute: " << groups;

                    std::size_t n_data_channels{data_shape.at(1)};
                    std::size_t n_filters_channels{weights_shape.at(0)};

                    ASSERT_VALID_ARGUMENT(node, n_data_channels % groups == 0)
                        << "provided group attribute value must be a multiple of data channels "
                           "count.";
                    ASSERT_VALID_ARGUMENT(node, n_filters_channels % groups == 0)
                        << "provided group attribute value must be a multiple of filter channels "
                           "count.";

                    std::shared_ptr<ngraph::Node> conv_node;
                    if (!output_shape.empty())
                    {
                        conv_node = std::make_shared<ngraph::opset0::GroupConvolutionTranspose>(
                            data,
                            filters,
                            strides,
                            dilations,
                            CoordinateDiff(std::begin(output_padding), std::end(output_padding)),
                            Shape(std::begin(output_shape), std::end(output_shape)),
                            groups);
                    }
                    else
                    {
                        conv_node = std::make_shared<ngraph::opset0::GroupConvolutionTranspose>(
                            data,
                            filters,
                            strides,
                            dilations,
                            padding_below,
                            padding_above,
                            CoordinateDiff(std::begin(output_padding), std::end(output_padding)),
                            groups,
                            auto_pad_type);
                    }

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }

                    auto bias = inputs.at(2);

                    return {std::make_shared<ngraph::opset0::Add>(
                        conv_node,
                        ngraph::op::make_broadcast_node(bias, conv_node->get_shape(), 1))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
