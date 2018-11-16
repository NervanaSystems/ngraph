//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include <iterator>
#include <memory>
#include <vector>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/frontend/onnx_import/op/conv_transpose.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
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

                    Shape data_batch_shape(data_shape.size(), 1);
                    data_batch_shape[0] = data_shape[0];
                    data_batch_shape[1] = weights_shape[1];

                    if (!output_shape.empty())
                    {
                        if (output_shape.size() > num_spatial_dims)
                        {
                            output_shape.erase(std::begin(output_shape),
                                               std::begin(output_shape) + 2);
                        }
                        for (int i = 0; i < num_spatial_dims; ++i)
                        {
                            padding_below[i] = (strides[i] * (data_shape[i + 2] - 1) +
                                                dilations[i] * (weights_shape[i + 2] - 1) -
                                                data_dilation_strides[i] *
                                                    (output_shape[i] - output_padding[i] - 1)) /
                                               2;
                            if (static_cast<int>(padding_below[i]) < 0)
                            {
                                output_padding[i] = -static_cast<int>(padding_below[i]);
                                padding_below[i] = 0;
                            }
                            padding_above[i] = padding_below[i];
                            data_batch_shape[i + 2] = output_shape[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < num_spatial_dims && output_shape.empty(); ++i)
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

                    auto conv_node = std::make_shared<ngraph::op::ConvolutionBackpropData>(
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
                    const Shape& new_shape = conv_node->get_shape();

                    return {std::make_shared<ngraph::op::Add>(
                        conv_node, make_broadcast_node(bias, conv_node->get_shape(), 1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
