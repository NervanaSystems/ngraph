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

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/quantize.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> ScaledQuantize(std::shared_ptr<Node> input,
                                             std::shared_ptr<Node> min,
                                             std::shared_ptr<Node> max,
                                             const ngraph::element::Type& type,
                                             const ngraph::AxisSet& axes,
                                             op::Quantize::RoundMode round_mode);

        std::shared_ptr<Node> ScaledDequantize(std::shared_ptr<Node> input,
                                               std::shared_ptr<Node> min,
                                               std::shared_ptr<Node> max,
                                               const ngraph::element::Type& type,
                                               const ngraph::AxisSet& axes);

        std::shared_ptr<Node> ScaledQuantizedAvgPool(std::shared_ptr<Node> input,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     bool include_padding_in_avg_computation,
                                                     std::shared_ptr<Node> min,
                                                     std::shared_ptr<Node> max);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionBias(std::shared_ptr<Node> input,
                                           std::shared_ptr<Node> filters,
                                           std::shared_ptr<Node> bias,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           std::shared_ptr<Node> min_input,
                                           std::shared_ptr<Node> max_input,
                                           std::shared_ptr<Node> min_filter,
                                           std::shared_ptr<Node> max_filter,
                                           std::shared_ptr<Node> min_freezed_output,
                                           std::shared_ptr<Node> max_freezed_output,
                                           const bool with_relu = false);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionRelu(std::shared_ptr<Node> input,
                                           std::shared_ptr<Node> filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           std::shared_ptr<Node> min_input,
                                           std::shared_ptr<Node> max_input,
                                           std::shared_ptr<Node> min_filter,
                                           std::shared_ptr<Node> max_filter,
                                           std::shared_ptr<Node> min_freezed_output,
                                           std::shared_ptr<Node> max_freezed_output);

        std::shared_ptr<Node> ScaledQuantizedConvolution(std::shared_ptr<Node> input,
                                                         std::shared_ptr<Node> filters,
                                                         const Strides& window_movement_strides,
                                                         const Strides& window_dilation_strides,
                                                         const CoordinateDiff& padding_below,
                                                         const CoordinateDiff& padding_above,
                                                         const Strides& data_dilation_strides,
                                                         std::shared_ptr<Node> min_input,
                                                         std::shared_ptr<Node> max_input,
                                                         std::shared_ptr<Node> min_filter,
                                                         std::shared_ptr<Node> max_filter,
                                                         std::shared_ptr<Node> min_freezed_output,
                                                         std::shared_ptr<Node> max_freezed_output);

        std::shared_ptr<Node> ScaledQuantizedMaxPool(std::shared_ptr<Node> input,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     std::shared_ptr<Node> min,
                                                     std::shared_ptr<Node> max);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionBiasAdd(std::shared_ptr<Node> input,
                                              std::shared_ptr<Node> filters,
                                              std::shared_ptr<Node> bias,
                                              std::shared_ptr<Node> sum_input,
                                              const Strides& window_movement_strides,
                                              const Strides& window_dilation_strides,
                                              const CoordinateDiff& padding_below,
                                              const CoordinateDiff& padding_above,
                                              const Strides& data_dilation_strides,
                                              std::shared_ptr<Node> min_input,
                                              std::shared_ptr<Node> max_input,
                                              std::shared_ptr<Node> min_filter,
                                              std::shared_ptr<Node> max_filter,
                                              std::shared_ptr<Node> min_freezed_output_conv_1,
                                              std::shared_ptr<Node> max_freezed_output_conv_1,
                                              std::shared_ptr<Node> min_freezed_output_conv_2,
                                              std::shared_ptr<Node> max_freezed_output_conv_2,
                                              const bool with_relu);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionBiasSignedAdd(std::shared_ptr<Node> input,
                                                    std::shared_ptr<Node> filters,
                                                    std::shared_ptr<Node> bias,
                                                    std::shared_ptr<Node> sum_input,
                                                    const Strides& window_movement_strides,
                                                    const Strides& window_dilation_strides,
                                                    const CoordinateDiff& padding_below,
                                                    const CoordinateDiff& padding_above,
                                                    const Strides& data_dilation_strides,
                                                    std::shared_ptr<Node> min_input,
                                                    std::shared_ptr<Node> max_input,
                                                    std::shared_ptr<Node> min_filter,
                                                    std::shared_ptr<Node> max_filter,
                                                    std::shared_ptr<Node> min_freezed_output_conv_1,
                                                    std::shared_ptr<Node> max_freezed_output_conv_1,
                                                    std::shared_ptr<Node> min_freezed_output_conv_2,
                                                    std::shared_ptr<Node> max_freezed_output_conv_2,
                                                    const bool with_relu);
        std::shared_ptr<Node>
            ScaledQuantizedConvolutionFusion(std::shared_ptr<Node> input,
                                             std::shared_ptr<Node> filters,
                                             std::shared_ptr<Node> bias,
                                             std::shared_ptr<ngraph::Node> gamma,
                                             std::shared_ptr<ngraph::Node> beta,
                                             std::shared_ptr<ngraph::Node> mean,
                                             std::shared_ptr<ngraph::Node> variance,
                                             std::shared_ptr<Node> sum_input,
                                             std::shared_ptr<Node> min_input,
                                             std::shared_ptr<Node> max_input,
                                             std::shared_ptr<Node> sum_min_input,
                                             std::shared_ptr<Node> sum_max_input,
                                             std::shared_ptr<Node> min_freezed_output_conv,
                                             std::shared_ptr<Node> max_freezed_output_conv,
                                             const Strides& window_movement_strides,
                                             const Strides& window_dilation_strides,
                                             const CoordinateDiff& padding_below,
                                             const CoordinateDiff& padding_above,
                                             const Strides& data_dilation_strides,
                                             const bool with_relu,
                                             const bool with_bn);
    }
}
