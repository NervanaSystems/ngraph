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
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/reshape.hpp"
#include "quantization_util.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        shared_ptr<Node> ScaledQuantize(const shared_ptr<Node>& input,
                                        const shared_ptr<Node>& min,
                                        const shared_ptr<Node>& max,
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

        shared_ptr<Node> ScaledDequantize(const shared_ptr<Node>& input,
                                          const shared_ptr<Node>& min,
                                          const shared_ptr<Node>& max,
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

        shared_ptr<Node> ScaledQuantizedConcat(const NodeVector& args,
                                               size_t concatenation_axis,
                                               const NodeVector& mins,
                                               const NodeVector& maxs)
        {
            quantization_util::check_concat(args, mins, maxs);
            auto quant_type = args[0]->get_element_type();

            // output scale
            auto min = make_shared<op::Min>(make_shared<op::Concat>(mins, 0), ngraph::AxisSet{0});
            auto max = make_shared<op::Max>(make_shared<op::Concat>(maxs, 0), ngraph::AxisSet{0});
            auto out_scale = quantization_util::get_scale(min, max, quant_type);

            NodeVector rescaled_args(args.size());
            for (size_t i = 0; i < args.size(); ++i)
            {
                auto q_type = args[i]->get_element_type();
                auto in_scale = make_shared<ngraph::op::Reshape>(
                    quantization_util::get_scale(mins[i], maxs[i], q_type), AxisVector{0}, Shape{});
                auto zero = make_constant(q_type, in_scale->get_shape(), 0);

                rescaled_args[i] =
                    make_shared<op::Dequantize>(args[i], in_scale, zero, element::f32, AxisSet{});
                rescaled_args[i] =
                    make_shared<op::Quantize>(rescaled_args[i],
                                              out_scale,
                                              zero,
                                              q_type,
                                              AxisSet{},
                                              op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
            }

            return make_shared<op::QuantizedConcat>(rescaled_args, concatenation_axis);
        }

        shared_ptr<Node> ScaledQuantizedAvgPool(const shared_ptr<Node>& input,
                                                const Shape& window_shape,
                                                const Strides& window_movement_strides,
                                                const Shape& padding_below,
                                                const Shape& padding_above,
                                                bool include_padding_in_avg_computation,
                                                const shared_ptr<Node>& min,
                                                const shared_ptr<Node>& max)
        {
            return make_shared<op::QuantizedAvgPool>(input,
                                                     window_shape,
                                                     window_movement_strides,
                                                     padding_below,
                                                     padding_above,
                                                     include_padding_in_avg_computation);
        }

        shared_ptr<Node> ScaledQuantizedConvolutionBias(const shared_ptr<Node>& input,
                                                        const shared_ptr<Node>& filters,
                                                        const shared_ptr<Node>& bias,
                                                        const Strides& window_movement_strides,
                                                        const Strides& window_dilation_strides,
                                                        const CoordinateDiff& padding_below,
                                                        const CoordinateDiff& padding_above,
                                                        const Strides& data_dilation_strides,
                                                        const shared_ptr<Node>& min_input,
                                                        const shared_ptr<Node>& max_input,
                                                        const shared_ptr<Node>& min_filter,
                                                        const shared_ptr<Node>& max_filter,
                                                        const shared_ptr<Node>& min_output,
                                                        const shared_ptr<Node>& max_output,
                                                        const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(
                min_input, max_input, min_filter, max_filter, min_output, max_output, output_et);

            auto mybias = bias;
            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }

            return make_shared<op::QuantizedConvolutionBias>(input,
                                                             filters,
                                                             mybias,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale,
                                                             with_relu);
        }

        shared_ptr<Node> ScaledQuantizedConvolutionRelu(const shared_ptr<Node>& input,
                                                        const shared_ptr<Node>& filters,
                                                        const Strides& window_movement_strides,
                                                        const Strides& window_dilation_strides,
                                                        const CoordinateDiff& padding_below,
                                                        const CoordinateDiff& padding_above,
                                                        const Strides& data_dilation_strides,
                                                        const shared_ptr<Node>& min_input,
                                                        const shared_ptr<Node>& max_input,
                                                        const shared_ptr<Node>& min_filter,
                                                        const shared_ptr<Node>& max_filter,
                                                        const shared_ptr<Node>& min_output,
                                                        const shared_ptr<Node>& max_output)
        {
            auto requantization_scale = quantization_util::get_scale(
                min_input, max_input, min_filter, max_filter, min_output, max_output, element::u8);

            return make_shared<op::QuantizedConvolutionRelu>(input,
                                                             filters,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale);
        }

        shared_ptr<Node> ScaledQuantizedMaxPool(const shared_ptr<Node>& input,
                                                const Shape& window_shape,
                                                const Strides& window_movement_strides,
                                                const Shape& padding_below,
                                                const Shape& padding_above,
                                                const shared_ptr<Node>& min,
                                                const shared_ptr<Node>& max)
        {
            return make_shared<op::QuantizedMaxPool>(
                input, window_shape, window_movement_strides, padding_below, padding_above);
        }

        shared_ptr<Node> ScaledQuantizedConvolutionBiasAdd(const shared_ptr<Node>& input,
                                                           const shared_ptr<Node>& filters,
                                                           const shared_ptr<Node>& bias,
                                                           const shared_ptr<Node>& sum_input,
                                                           const Strides& window_movement_strides,
                                                           const Strides& window_dilation_strides,
                                                           const CoordinateDiff& padding_below,
                                                           const CoordinateDiff& padding_above,
                                                           const Strides& data_dilation_strides,
                                                           const shared_ptr<Node>& min_input,
                                                           const shared_ptr<Node>& max_input,
                                                           const shared_ptr<Node>& min_filter,
                                                           const shared_ptr<Node>& max_filter,
                                                           const shared_ptr<Node>& min_output,
                                                           const shared_ptr<Node>& max_output,
                                                           const shared_ptr<Node>& min_sum_input,
                                                           const shared_ptr<Node>& max_sum_input,
                                                           const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(
                min_input, max_input, min_filter, max_filter, min_output, max_output, output_et);

            auto sum_scale = builder::quantization_util::get_sum_scale(
                min_output, max_output, min_sum_input, max_sum_input);

            auto mybias = bias;
            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }

            return make_shared<op::QuantizedConvolutionBiasAdd>(input,
                                                                filters,
                                                                mybias,
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

        shared_ptr<Node>
            ScaledQuantizedConvolutionBiasSignedAdd(const shared_ptr<Node>& input,
                                                    const shared_ptr<Node>& filters,
                                                    const shared_ptr<Node>& bias,
                                                    const shared_ptr<Node>& sum_input,
                                                    const Strides& window_movement_strides,
                                                    const Strides& window_dilation_strides,
                                                    const CoordinateDiff& padding_below,
                                                    const CoordinateDiff& padding_above,
                                                    const Strides& data_dilation_strides,
                                                    const shared_ptr<Node>& min_input,
                                                    const shared_ptr<Node>& max_input,
                                                    const shared_ptr<Node>& min_filter,
                                                    const shared_ptr<Node>& max_filter,
                                                    const shared_ptr<Node>& min_output,
                                                    const shared_ptr<Node>& max_output,
                                                    const shared_ptr<Node>& min_sum_input,
                                                    const shared_ptr<Node>& max_sum_input,
                                                    const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto requantization_scale = quantization_util::get_scale(
                min_input, max_input, min_filter, max_filter, min_output, max_output, output_et);

            auto sum_scale = builder::quantization_util::get_sum_scale(
                min_output, max_output, min_sum_input, max_sum_input);
            if (output_et == element::u8)
            {
                // Need to multiply by two to account for u8 requantization_scale
                auto two = make_constant(element::f32, sum_scale->get_shape(), 2.0f);
                sum_scale = two * sum_scale;
            }

            auto mybias = bias;
            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            auto qconv = make_shared<op::QuantizedConvolutionBiasSignedAdd>(input,
                                                                            filters,
                                                                            mybias,
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

        shared_ptr<Node> ScaledQuantizedDotBias(const shared_ptr<Node>& input,
                                                const shared_ptr<Node>& filters,
                                                const shared_ptr<Node>& bias,
                                                const shared_ptr<Node>& min_input,
                                                const shared_ptr<Node>& max_input,
                                                const shared_ptr<Node>& min_filter,
                                                const shared_ptr<Node>& max_filter,
                                                const shared_ptr<Node>& min_output,
                                                const shared_ptr<Node>& max_output,
                                                const bool requantize,
                                                const bool with_relu)
        {
            auto requantization_scale =
                quantization_util::get_dot_scale(min_input,
                                                 max_input,
                                                 min_filter,
                                                 max_filter,
                                                 min_output,
                                                 max_output,
                                                 input->get_element_type(),
                                                 with_relu ? element::u8 : element::i8,
                                                 requantize);

            auto mybias = bias;
            if (bias->get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input->get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale =
                    quantization_util::get_bias_scale(min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            return make_shared<op::QuantizedDotBias>(
                input, filters, mybias, requantization_scale, requantize, with_relu);
        }

        shared_ptr<Node> ScaledQuantizedDot(const shared_ptr<Node>& input,
                                            const shared_ptr<Node>& filters,
                                            const shared_ptr<Node>& min_input,
                                            const shared_ptr<Node>& max_input,
                                            const shared_ptr<Node>& min_filter,
                                            const shared_ptr<Node>& max_filter,
                                            const shared_ptr<Node>& min_output,
                                            const shared_ptr<Node>& max_output,
                                            const bool requantize,
                                            const bool with_relu)
        {
            auto requantization_scale =
                quantization_util::get_dot_scale(min_input,
                                                 max_input,
                                                 min_filter,
                                                 max_filter,
                                                 min_output,
                                                 max_output,
                                                 input->get_element_type(),
                                                 with_relu ? element::u8 : element::i8,
                                                 requantize);
            return make_shared<op::QuantizedDot>(
                input, filters, requantization_scale, requantize, with_relu);
        }
    } // namespace builder
} // namespace ngraph
