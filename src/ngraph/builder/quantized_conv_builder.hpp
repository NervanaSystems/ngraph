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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node>
            QuantizedConvolutionBuilder(const NodeOutput& input,
                                        const NodeOutput& filters,
                                        const Strides& window_movement_strides,
                                        const Strides& window_dilation_strides,
                                        const CoordinateDiff& padding_below,
                                        const CoordinateDiff& padding_above,
                                        const Strides& data_dilation_strides,
                                        const NodeOutput& min_input,
                                        const NodeOutput& max_input,
                                        const NodeOutput& min_filter,
                                        const NodeOutput& max_filter,
                                        const NodeOutput& min_output,
                                        const NodeOutput& max_output,
                                        const ngraph::element::Type& output_type,
                                        const ngraph::AxisSet& input_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& filter_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& output_axes = ngraph::AxisSet{});

        std::shared_ptr<Node>
            QuantizedConvolutionBiasBuilder(const NodeOutput& input,
                                            const NodeOutput& filters,
                                            const NodeOutput& bias,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& data_dilation_strides,
                                            const NodeOutput& min_input,
                                            const NodeOutput& max_input,
                                            const NodeOutput& min_filter,
                                            const NodeOutput& max_filter,
                                            const NodeOutput& min_output,
                                            const NodeOutput& max_output,
                                            const bool with_relu = false);

        std::shared_ptr<Node>
            QuantizedConvolutionReluBuilder(const NodeOutput& input,
                                            const NodeOutput& filters,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& data_dilation_strides,
                                            const NodeOutput& min_input,
                                            const NodeOutput& max_input,
                                            const NodeOutput& min_filter,
                                            const NodeOutput& max_filter,
                                            const NodeOutput& min_output,
                                            const NodeOutput& max_output);

        std::shared_ptr<Node>
            QuantizedConvolutionBiasAddBuilder(const NodeOutput& input,
                                               const NodeOutput& filters,
                                               const NodeOutput& bias,
                                               const NodeOutput& sum_input,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const NodeOutput& min_input,
                                               const NodeOutput& max_input,
                                               const NodeOutput& min_filter,
                                               const NodeOutput& max_filter,
                                               const NodeOutput& min_output,
                                               const NodeOutput& max_output,
                                               const NodeOutput& min_sum_input,
                                               const NodeOutput& max_sum_input,
                                               const bool with_relu = false);

        std::shared_ptr<Node>
            QuantizedConvolutionBiasSignedAddBuilder(const NodeOutput& input,
                                                     const NodeOutput& filters,
                                                     const NodeOutput& bias,
                                                     const NodeOutput& sum_input,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const CoordinateDiff& padding_below,
                                                     const CoordinateDiff& padding_above,
                                                     const Strides& data_dilation_strides,
                                                     const NodeOutput& min_input,
                                                     const NodeOutput& max_input,
                                                     const NodeOutput& min_filter,
                                                     const NodeOutput& max_filter,
                                                     const NodeOutput& min_output,
                                                     const NodeOutput& max_output,
                                                     const NodeOutput& min_sum_input,
                                                     const NodeOutput& max_sum_input,
                                                     const bool with_relu = false);
    }
}
