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

        std::shared_ptr<Node> ScaledQuantizedAvgPool(const std::shared_ptr<Node>& arg,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     bool include_padding_in_avg_computation,
                                                     const std::shared_ptr<Node> min,
                                                     const std::shared_ptr<Node> max);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionBias(const std::shared_ptr<Node>& data_batch,
                                           const std::shared_ptr<Node>& filters,
                                           const std::shared_ptr<Node>& bias,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const std::shared_ptr<Node> min_input,
                                           const std::shared_ptr<Node> max_input,
                                           const std::shared_ptr<Node> min_filter,
                                           const std::shared_ptr<Node> max_filter,
                                           const std::shared_ptr<Node> min_freezed_output,
                                           const std::shared_ptr<Node> max_freezed_output,
                                           const bool with_relu = false);

        std::shared_ptr<Node>
            ScaledQuantizedConvolutionRelu(const std::shared_ptr<Node>& data_batch,
                                           const std::shared_ptr<Node>& filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const std::shared_ptr<Node> min_input,
                                           const std::shared_ptr<Node> max_input,
                                           const std::shared_ptr<Node> min_filter,
                                           const std::shared_ptr<Node> max_filter,
                                           const std::shared_ptr<Node> min_freezed_output,
                                           const std::shared_ptr<Node> max_freezed_output);

        std::shared_ptr<Node>
            ScaledQuantizedConvolution(const std::shared_ptr<Node>& data_batch,
                                       const std::shared_ptr<Node>& filters,
                                       const Strides& window_movement_strides,
                                       const Strides& window_dilation_strides,
                                       const CoordinateDiff& padding_below,
                                       const CoordinateDiff& padding_above,
                                       const Strides& data_dilation_strides,
                                       const std::shared_ptr<Node> min_input,
                                       const std::shared_ptr<Node> max_input,
                                       const std::shared_ptr<Node> min_filter,
                                       const std::shared_ptr<Node> max_filter,
                                       const std::shared_ptr<Node> min_freezed_output,
                                       const std::shared_ptr<Node> max_freezed_output);

        std::shared_ptr<Node> ScaledQuantizedMaxPool(const std::shared_ptr<Node>& arg,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     const std::shared_ptr<Node> min,
                                                     const std::shared_ptr<Node> max);
    }
}
