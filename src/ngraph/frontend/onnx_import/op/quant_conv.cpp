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

#include <cstddef>
#include <memory>
#include <vector>

#include "ngraph/builder/quantization/quantized_linear_convolution.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/op/conv.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/slice.hpp"
#include "quant_conv.hpp"

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
                    struct OpScale
                    {
                        std::shared_ptr<ngraph::Node> data_scale;
                        std::shared_ptr<ngraph::Node> filter_scale;
                        std::shared_ptr<ngraph::Node> output_scale;
                    };

                    struct OpZeroPoint
                    {
                        std::shared_ptr<ngraph::Node> data_zero_point;
                        std::shared_ptr<ngraph::Node> filter_zero_point;
                        std::shared_ptr<ngraph::Node> output_zero_point;
                    };

                    std::shared_ptr<ngraph::Node>
                        make_ng_quant_conv(const std::shared_ptr<ngraph::Node>& data,
                                           const std::shared_ptr<ngraph::Node>& filters,
                                           const ngraph::Strides& strides,
                                           const ngraph::Strides& filter_dilations,
                                           const ngraph::CoordinateDiff& padding_below,
                                           const ngraph::CoordinateDiff& padding_above,
                                           const ngraph::Strides& data_dilations,
                                           int groups,
                                           const OpScale& op_scale,
                                           const OpZeroPoint& op_zero_point)
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
                            ngraph::NodeVector convolution_nodes;

                            // initial bounds for splice
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

                                convolution_nodes.push_back(
                                    ngraph::builder::quantization::QuantizedLinearConvolution(
                                        sliced_data,
                                        sliced_filters,
                                        strides,
                                        filter_dilations,
                                        padding_below,
                                        padding_above,
                                        data_dilations,
                                        op_scale.data_scale,
                                        op_scale.filter_scale,
                                        op_scale.output_scale));
                            }
                            std::size_t concatenation_axis = 1;
                            return std::make_shared<ngraph::op::Concat>(convolution_nodes,
                                                                        concatenation_axis);
                        }
                        else
                        {
                            return ngraph::builder::quantization::QuantizedLinearConvolution(
                                data,
                                filters,
                                strides,
                                filter_dilations,
                                padding_below,
                                padding_above,
                                data_dilations,
                                op_scale.data_scale,
                                op_scale.filter_scale,
                                op_scale.output_scale);
                        }
                    }

                    std::shared_ptr<ngraph::Node>
                        make_ng_quant_conv_bias(const std::shared_ptr<ngraph::Node>& data,
                                                const std::shared_ptr<ngraph::Node>& filters,
                                                const std::shared_ptr<ngraph::Node>& bias,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& filter_dilations,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above,
                                                const ngraph::Strides& data_dilations,
                                                int groups,
                                                const OpScale& op_scale,
                                                const OpZeroPoint& op_zero_point)
                    {
                        if (groups > 1)
                        {
                            throw ngraph::onnx_import::error::NotSupported(
                                "No support for quantized group conv+bias.");
                        }
                        else
                        {
                            return ngraph::builder::quantization::QuantizedLinearConvolutionBias(
                                data,
                                filters,
                                bias,
                                strides,
                                filter_dilations,
                                padding_below,
                                padding_above,
                                data_dilations,
                                op_scale.data_scale,
                                op_scale.filter_scale,
                                op_scale.output_scale);
                        }
                    }

                } // namespace

                ngraph::NodeVector quant_conv(const ngraph::onnx_import::Node& node)
                {
                    const ngraph::NodeVector& inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    auto filters = inputs.at(3);

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    auto data_zp = inputs.at(2);
                    auto filters_zp = inputs.at(5);
                    auto output_zp = inputs.at(7);

                    auto data_scale = inputs.at(1);
                    auto filters_scale = inputs.at(4);
                    auto output_scale = inputs.at(6);

                    auto scale = data_scale * filters_scale / output_scale;

                    ASSERT_VALID_ARGUMENT(node,
                                          ((groups >= 0) && (groups <= data->get_shape().at(1)) &&
                                           (groups <= filters->get_shape().at(0))))
                        << "incorrect value of 'group' attribute: " << groups;

                    ngraph::Strides strides = ngraph::onnx_import::convpool::get_strides(node);
                    ngraph::Strides filter_dilations =
                        ngraph::onnx_import::convpool::get_dilations(node);
                    ngraph::Strides data_dilations = ngraph::Strides(
                        ngraph::onnx_import::convpool::get_kernel_shape(node).size(), 1UL);
                    auto paddings = ngraph::onnx_import::convpool::get_pads(node);
                    const ngraph::CoordinateDiff& padding_below = paddings.first;
                    const ngraph::CoordinateDiff& padding_above = paddings.second;

                    std::shared_ptr<ngraph::Node> conv_node = nullptr;

                    // no bias param
                    if (inputs.size() < 9)
                    {
                        conv_node =
                            make_ng_quant_conv(data,
                                               filters,
                                               strides,
                                               filter_dilations,
                                               padding_below,
                                               padding_above,
                                               data_dilations,
                                               groups,
                                               OpScale{data_scale, filters_scale, output_scale},
                                               OpZeroPoint{data_zp, filters_zp, output_zp});
                    }
                    else
                    {
                        auto bias = inputs.at(8);
                        conv_node = make_ng_quant_conv_bias(
                            data,
                            filters,
                            bias,
                            strides,
                            filter_dilations,
                            padding_below,
                            padding_above,
                            data_dilations,
                            groups,
                            OpScale{data_scale, filters_scale, output_scale},
                            OpZeroPoint{data_zp, filters_zp, output_zp});
                    }

                    return {conv_node};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph