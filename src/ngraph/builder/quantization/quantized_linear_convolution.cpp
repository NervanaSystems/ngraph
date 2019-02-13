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
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            shared_ptr<Node> QuantizedLinearConvolution(shared_ptr<Node> input,
                                                        shared_ptr<Node> filter,
                                                        const Strides& window_movement_strides,
                                                        const Strides& window_dilation_strides,
                                                        const CoordinateDiff& padding_below,
                                                        const CoordinateDiff& padding_above,
                                                        const Strides& data_dilation_strides,
                                                        shared_ptr<Node> input_scale,
                                                        shared_ptr<Node> filter_scale,
                                                        shared_ptr<Node> output_scale)
            {
                // TODO: need to establish cross-nGraph view of scale (mult or div)
                auto requantization_scale = (input_scale * filter_scale) / output_scale;

                return make_shared<op::QuantizedConvolution>(input,
                                                             filter,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale);
            }

            shared_ptr<Node> QuantizedLinearConvolutionBias(shared_ptr<Node> input,
                                                            shared_ptr<Node> filter,
                                                            shared_ptr<Node> bias,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            const CoordinateDiff& padding_below,
                                                            const CoordinateDiff& padding_above,
                                                            const Strides& data_dilation_strides,
                                                            shared_ptr<Node> input_scale,
                                                            shared_ptr<Node> filter_scale,
                                                            shared_ptr<Node> output_scale)
            {
                // TODO: need to establish cross-nGraph view of scale (mult or div)
                auto requantization_scale = (input_scale * filter_scale) / output_scale;

                if (bias->get_element_type() != element::i32)
                {
                    auto zero = make_constant(element::i32, input_scale->get_shape(), 0);
                    AxisSet quantization_axes;
                    auto bias_scale = input_scale * filter_scale;
                    op::Quantize::RoundMode round_mode =
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                    bias = make_shared<op::Quantize>(
                        bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
                }

                return make_shared<op::QuantizedConvolutionBias>(input,
                                                                 filter,
                                                                 bias,
                                                                 window_movement_strides,
                                                                 window_dilation_strides,
                                                                 padding_below,
                                                                 padding_above,
                                                                 data_dilation_strides,
                                                                 requantization_scale,
                                                                 false);
            }
        }
    }
}
