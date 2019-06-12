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

#include "ngraph/builder/quantization/quantized_linear_convolution.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            shared_ptr<Node> QuantizedLinearConvolution(const shared_ptr<Node>& input,
                                                        const shared_ptr<Node>& filter,
                                                        const Strides& window_movement_strides,
                                                        const Strides& window_dilation_strides,
                                                        const CoordinateDiff& padding_below,
                                                        const CoordinateDiff& padding_above,
                                                        const Strides& data_dilation_strides,
                                                        const shared_ptr<Node>& input_scale,
                                                        const shared_ptr<Node>& filter_scale,
                                                        const shared_ptr<Node>& output_scale)
            {
                // TODO: need to establish cross-nGraph view of scale (mult or div)
                const auto requantization_scale = (input_scale * filter_scale) / output_scale;

                return make_shared<op::QuantizedConvolution>(input,
                                                             filter,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale);
            }

            // TODO: this codes is falling back to fp32 convolution
            //       need to make this the primary builder which means
            //       1) add support for zero point in QuantizeConvolution op API
            //       2) add QuantizedConvolution reference kernel, including zero point
            shared_ptr<Node> QuantizedLinearConvolution(const shared_ptr<Node>& input,
                                                        const shared_ptr<Node>& filter,
                                                        const Strides& window_movement_strides,
                                                        const Strides& window_dilation_strides,
                                                        const CoordinateDiff& padding_below,
                                                        const CoordinateDiff& padding_above,
                                                        const Strides& data_dilation_strides,
                                                        const shared_ptr<Node>& input_scale,
                                                        const shared_ptr<Node>& input_zero_point,
                                                        const shared_ptr<Node>& filter_scale,
                                                        const shared_ptr<Node>& filter_zero_point,
                                                        const shared_ptr<Node>& output_scale,
                                                        const shared_ptr<Node>& output_zero_point)
            {
                const AxisSet axes;

                const auto dq_input = make_shared<op::Dequantize>(
                    input, input_scale, input_zero_point, input_scale->get_element_type(), axes);

                const auto dq_filter = make_shared<op::Dequantize>(filter,
                                                                   filter_scale,
                                                                   filter_zero_point,
                                                                   filter_scale->get_element_type(),
                                                                   axes);

                const auto convolution = make_shared<op::Convolution>(dq_input,
                                                                      dq_filter,
                                                                      window_movement_strides,
                                                                      window_dilation_strides,
                                                                      padding_below,
                                                                      padding_above,
                                                                      data_dilation_strides);
                // Return quantized convolution
                return make_shared<op::Quantize>(
                    convolution,
                    output_scale,
                    output_zero_point,
                    output_zero_point->get_element_type(),
                    axes,
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
            }

            shared_ptr<Node> QuantizedLinearConvolutionBias(const shared_ptr<Node>& input,
                                                            const shared_ptr<Node>& filter,
                                                            const shared_ptr<Node>& bias,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            const CoordinateDiff& padding_below,
                                                            const CoordinateDiff& padding_above,
                                                            const Strides& data_dilation_strides,
                                                            const shared_ptr<Node>& input_scale,
                                                            const shared_ptr<Node>& filter_scale,
                                                            const shared_ptr<Node>& output_scale)
            {
                // TODO: need to establish cross-nGraph view of scale (mult or div)
                auto requantization_scale = (input_scale * filter_scale) / output_scale;

                auto mybias = bias;
                if (bias->get_element_type() != element::i32)
                {
                    const auto zero = make_constant(element::i32, input_scale->get_shape(), 0);
                    const AxisSet quantization_axes;
                    const auto bias_scale = input_scale * filter_scale;
                    op::Quantize::RoundMode round_mode =
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                    mybias = make_shared<op::Quantize>(
                        bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
                }

                return make_shared<op::QuantizedConvolutionBias>(input,
                                                                 filter,
                                                                 mybias,
                                                                 window_movement_strides,
                                                                 window_dilation_strides,
                                                                 padding_below,
                                                                 padding_above,
                                                                 data_dilation_strides,
                                                                 requantization_scale,
                                                                 false);
            }

            shared_ptr<Node> QuantizedConvInteger(const shared_ptr<Node>& input,
                                                  const shared_ptr<Node>& filter,
                                                  const Strides& window_movement_strides,
                                                  const Strides& window_dilation_strides,
                                                  const CoordinateDiff& padding_below,
                                                  const CoordinateDiff& padding_above,
                                                  const Strides& data_dilation_strides)
            {
                auto output_scale = make_constant(element::f32, Shape{}, 1);
                return make_shared<op::QuantizedConvolution>(input,
                                                             filter,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             output_scale,
                                                             false);
            }

            shared_ptr<Node> QuantizedConvInteger(const shared_ptr<Node>& input,
                                                  const shared_ptr<Node>& filter,
                                                  const Strides& window_movement_strides,
                                                  const Strides& window_dilation_strides,
                                                  const CoordinateDiff& padding_below,
                                                  const CoordinateDiff& padding_above,
                                                  const Strides& data_dilation_strides,
                                                  const std::shared_ptr<Node>& input_zero_point,
                                                  const std::shared_ptr<Node>& filter_zero_point)
            {
                // Check if zero points are constant and zero
                if (ngraph::is_zero(input_zero_point) && ngraph::is_zero(filter_zero_point))
                {
                    return QuantizedConvInteger(input,
                                                filter,
                                                window_movement_strides,
                                                window_dilation_strides,
                                                padding_below,
                                                padding_above,
                                                data_dilation_strides);
                }
                else
                {
                    // Fall back to performing operation on dequantized floating-point values
                    const auto input_scale = make_constant(element::f32, Shape{}, 1);
                    const auto filter_scale = make_constant(element::f32, Shape{}, 1);
                    const auto output_scale = make_constant(element::f32, Shape{}, 1);
                    const auto output_zero_point = make_constant(element::i32, Shape{}, 0);
                    const AxisSet axes;

                    const auto dq_input =
                        make_shared<op::Dequantize>(input,
                                                    input_scale,
                                                    input_zero_point,
                                                    input_scale->get_element_type(),
                                                    axes);

                    const auto dq_filter =
                        make_shared<op::Dequantize>(filter,
                                                    filter_scale,
                                                    filter_zero_point,
                                                    filter_scale->get_element_type(),
                                                    axes);

                    const auto output = make_shared<op::Convolution>(dq_input,
                                                                     dq_filter,
                                                                     window_movement_strides,
                                                                     window_dilation_strides,
                                                                     padding_below,
                                                                     padding_above,
                                                                     data_dilation_strides);

                    // Return quantized output
                    return make_shared<op::Quantize>(
                        output,
                        output_scale,
                        output_zero_point,
                        output_zero_point->get_element_type(),
                        axes,
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
                }
            }
        }
    }
}
