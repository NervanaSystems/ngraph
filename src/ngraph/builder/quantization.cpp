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

#include <memory>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "quantization_util.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> ScaledQuantize(std::shared_ptr<Node> input,
                                             std::shared_ptr<Node> min,
                                             std::shared_ptr<Node> max,
                                             const ngraph::element::Type& quant_type,
                                             const ngraph::AxisSet& axes,
                                             op::Quantize::RoundMode round_mode)
        {
            auto real_type = input->get_element_type();

            if (min->get_element_type() != real_type)
            {
                throw ngraph_error("ScaledQuantize: min must match input type");
            }

            if (max->get_element_type() != real_type)
            {
                throw ngraph_error("ScaledQuantize: max must match input type");
            }

            auto shape = min->get_shape();
            if (shape != max->get_shape())
            {
                throw ngraph_error("ScaledQuantize: min and max must have same shape");
            }

            auto zero = make_constant(quant_type, shape, 0);
            auto scale = quantization_util::get_scale(min, max, quant_type, true);
            return make_shared<op::Quantize>(input, scale, zero, quant_type, axes, round_mode);
        }

        std::shared_ptr<Node> ScaledDequantize(std::shared_ptr<Node> input,
                                               std::shared_ptr<Node> min,
                                               std::shared_ptr<Node> max,
                                               const ngraph::element::Type& real_type,
                                               const ngraph::AxisSet& axes)
        {
            auto quant_type = input->get_element_type();

            if (min->get_element_type() != real_type)
            {
                throw ngraph_error("ScaledDequantize: min must match output type");
            }

            if (max->get_element_type() != real_type)
            {
                throw ngraph_error("ScaledDequantize: max must match output type");
            }

            auto shape = min->get_shape();
            if (shape != max->get_shape())
            {
                throw ngraph_error("ScaledDequantize: min and max must have same shape");
            }

            auto zero = make_constant(quant_type, shape, 0);
            auto scale = quantization_util::get_scale(min, max, quant_type);
            return make_shared<op::Dequantize>(input, scale, zero, real_type, axes);
        }

        std::shared_ptr<Node> ScaledQuantizedAvgPool(std::shared_ptr<Node> input,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     bool include_padding_in_avg_computation,
                                                     std::shared_ptr<Node> min,
                                                     std::shared_ptr<Node> max)
        {
            return make_shared<op::QuantizedAvgPool>(input,
                                                     window_shape,
                                                     window_movement_strides,
                                                     padding_below,
                                                     padding_above,
                                                     include_padding_in_avg_computation);
        }

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
                                           const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(min_input,
                                                                     max_input,
                                                                     min_filter,
                                                                     max_filter,
                                                                     min_freezed_output,
                                                                     max_freezed_output,
                                                                     output_et);

            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                bias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            return make_shared<op::QuantizedConvolutionBias>(input,
                                                             filters,
                                                             bias,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale,
                                                             with_relu);
        }

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
                                           std::shared_ptr<Node> max_freezed_output)
        {
            auto requantization_scale = quantization_util::get_scale(min_input,
                                                                     max_input,
                                                                     min_filter,
                                                                     max_filter,
                                                                     min_freezed_output,
                                                                     max_freezed_output,
                                                                     element::u8);

            return make_shared<op::QuantizedConvolutionRelu>(input,
                                                             filters,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale);
        }

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
                                                         std::shared_ptr<Node> max_freezed_output)
        {
            auto requantization_scale = quantization_util::get_scale(min_input,
                                                                     max_input,
                                                                     min_filter,
                                                                     max_filter,
                                                                     min_freezed_output,
                                                                     max_freezed_output,
                                                                     element::i8);

            return make_shared<op::QuantizedConvolution>(input,
                                                         filters,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         requantization_scale);
        }

        std::shared_ptr<Node> ScaledQuantizedMaxPool(std::shared_ptr<Node> input,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     std::shared_ptr<Node> min,
                                                     std::shared_ptr<Node> max)
        {
            return make_shared<op::QuantizedMaxPool>(
                input, window_shape, window_movement_strides, padding_below, padding_above);
        }

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
                                              const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(min_input,
                                                                     max_input,
                                                                     min_filter,
                                                                     max_filter,
                                                                     min_freezed_output_conv_1,
                                                                     max_freezed_output_conv_1,
                                                                     output_et);

            auto sum_scale = builder::quantization_util::get_sum_scale(min_freezed_output_conv_1,
                                                                       max_freezed_output_conv_1,
                                                                       min_freezed_output_conv_2,
                                                                       max_freezed_output_conv_2);

            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                bias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            return make_shared<op::QuantizedConvolutionBiasAdd>(input,
                                                                filters,
                                                                bias,
                                                                sum_input,
                                                                window_movement_strides,
                                                                window_dilation_strides,
                                                                padding_below,
                                                                padding_above,
                                                                data_dilation_strides,
                                                                requantization_scale,
                                                                sum_scale,
                                                                with_relu);
        }

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
                                                    const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(min_input,
                                                                     max_input,
                                                                     min_filter,
                                                                     max_filter,
                                                                     min_freezed_output_conv_1,
                                                                     max_freezed_output_conv_1,
                                                                     output_et);

            auto sum_scale = builder::quantization_util::get_sum_scale(min_freezed_output_conv_1,
                                                                       max_freezed_output_conv_1,
                                                                       min_freezed_output_conv_2,
                                                                       max_freezed_output_conv_2);
            if (output_et == element::u8)
            {
                // Need to multiply by two to account for u8 requantization_scale
                auto two = make_constant(element::f32, sum_scale->get_shape(), 2.0f);
                sum_scale = two * sum_scale;
            }

            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                bias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            auto qconv = make_shared<op::QuantizedConvolutionBiasSignedAdd>(input,
                                                                            filters,
                                                                            bias,
                                                                            sum_input,
                                                                            window_movement_strides,
                                                                            window_dilation_strides,
                                                                            padding_below,
                                                                            padding_above,
                                                                            data_dilation_strides,
                                                                            requantization_scale,
                                                                            sum_scale,
                                                                            with_relu);
            return make_shared<op::Convert>(qconv, element::u8);
        }
    }
}
