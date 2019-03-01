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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/op/conv_transpose.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

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
                    std::shared_ptr<ngraph::Node>
                        make_ng_conv_transpose(std::int64_t groups,
                                               const Shape& data_batch_shape,
                                               const std::shared_ptr<ngraph::Node>& filters,
                                               const std::shared_ptr<ngraph::Node>& data,
                                               const Strides& strides,
                                               const Strides& dilations,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides)
                    {
                        if (groups > 1)
                        {
                            // Split one convolution op to N ops where N is the number of groups
                            // and concat results after computation.
                            std::size_t n_data_channels{data->get_shape().at(1)};
                            std::size_t n_filters_channels{filters->get_shape().at(0)};
                            std::size_t data_group_size{n_data_channels / groups};
                            std::size_t filters_group_size{n_filters_channels / groups};
                            NodeVector conv_transpose_nodes;

                            // initial bounds for slice
                            std::vector<std::size_t> data_lower_bounds(data->get_shape().size());
                            std::vector<std::size_t> data_upper_bounds{data->get_shape()};
                            std::vector<std::size_t> filters_lower_bounds(
                                filters->get_shape().size());
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

                                conv_transpose_nodes.push_back(
                                    std::make_shared<ngraph::op::ConvolutionBackpropData>(
                                        data_batch_shape,
                                        sliced_filters,
                                        sliced_data,
                                        strides,
                                        dilations,
                                        padding_below,
                                        padding_above,
                                        data_dilation_strides));
                            }
                            std::size_t concatenation_axis = 1;
                            return std::make_shared<ngraph::op::Concat>(conv_transpose_nodes,
                                                                        concatenation_axis);
                        }
                        else
                        {
                            return std::make_shared<ngraph::op::ConvolutionBackpropData>(
                                data_batch_shape,
                                filters,
                                data,
                                strides,
                                dilations,
                                padding_below,
                                padding_above,
                                data_dilation_strides);
                        }
                    }
                } // anonymous namespace

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
                    ngraph::CoordinateDiff padding_below = paddings.first;
                    ngraph::CoordinateDiff padding_above = paddings.second;

                    Strides data_dilation_strides(num_spatial_dims, 1);
                    std::vector<std::int64_t> output_shape{
                        node.get_attribute_value<std::vector<std::int64_t>>("output_shape", {})};

                    std::vector<std::int64_t> output_padding{
                        node.get_attribute_value<std::vector<std::int64_t>>(
                            "output_padding", std::vector<std::int64_t>(num_spatial_dims, 0))};

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    ASSERT_VALID_ARGUMENT(node,
                                          ((groups >= 0) && (groups <= data->get_shape().at(1)) &&
                                           (groups <= filters->get_shape().at(0))))
                        << "incorrect value of 'group' attribute: " << groups;

                    std::size_t n_data_channels{data_shape.at(1)};
                    std::size_t n_filters_channels{weights_shape.at(0)};

                    ASSERT_VALID_ARGUMENT(node, n_data_channels % groups == 0)
                        << "provided group attribute value must be a multiple of data channels "
                           "count.";
                    ASSERT_VALID_ARGUMENT(node, n_filters_channels % groups == 0)
                        << "provided group attribute value must be a multiple of filter channels "
                           "count.";

                    Shape data_batch_shape(data_shape.size(), 1);
                    data_batch_shape.at(0) = data_shape.at(0);
                    data_batch_shape.at(1) = weights_shape.at(1);

                    if (!output_shape.empty())
                    {
                        if (output_shape.size() > num_spatial_dims)
                        {
                            output_shape.erase(std::begin(output_shape),
                                               std::begin(output_shape) + 2);
                        }
                        for (int i = 0; i < num_spatial_dims; ++i)
                        {
                            padding_below[i] = strides[i] * (data_shape[i + 2] - 1) +
                                               dilations[i] * (weights_shape[i + 2] - 1) -
                                               data_dilation_strides[i] *
                                                   (output_shape[i] - output_padding[i] - 1);
                            if (padding_below[i] < 0)
                            {
                                // (int) -9 / 2 = -5 but we need -4
                                // (int) -9 --> 9 / 2 = 4 --> -4
                                padding_below[i] = -(-padding_below[i] / 2);
                            }
                            else
                            {
                                padding_below[i] /= 2;
                            }
                            padding_above[i] = padding_below[i];
                            data_batch_shape[i + 2] = output_shape[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < num_spatial_dims; ++i)
                        {
                            // Calculating spatial dims of data output shape for ngraph conv backprop op
                            // | s(ds-1) + d(ws-1) - pb - pa |
                            // | --------------------------- | + 1 + op
                            // | _           dds           _ |
                            //
                            // d - dilation
                            // ds - data shape
                            // dds - data dilation strides
                            // op - output padding
                            // pa - padding above
                            // pb - padding below
                            // s - strides
                            // ws - weights shape
                            data_batch_shape[i + 2] = (strides[i] * (data_shape[i + 2] - 1) +
                                                       dilations[i] * (weights_shape[i + 2] - 1) -
                                                       padding_below[i] - padding_above[i]) /
                                                          data_dilation_strides[i] +
                                                      1 + output_padding[i];
                        }
                    }

                    std::shared_ptr<ngraph::Node> conv_node =
                        make_ng_conv_transpose(groups,
                                               data_batch_shape,
                                               filters,
                                               data,
                                               strides,
                                               dilations,
                                               padding_below,
                                               padding_above,
                                               data_dilation_strides);

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }

                    auto bias = inputs.at(2);

                    return {std::make_shared<ngraph::op::Add>(
                        conv_node, make_broadcast_node(bias, conv_node->get_shape(), 1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
