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

#include <memory>

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
                min = make_shared<op::Convert>(min, real_type);
            }

            if (max->get_element_type() != real_type)
            {
                max = make_shared<op::Convert>(max, real_type);
            }

            auto shape = min->get_shape();
            if (shape != max->get_shape())
            {
                throw ngraph_error("ScaledQuantize: min and max must have same shape");
            }

            auto zero = op::Constant::create(quant_type, shape, {0});
            auto scale =
                builder::quantization_util::get_quantization_scale(min, max, quant_type, true);
            return make_shared<op::Quantize>(input, scale, zero, quant_type, axes, round_mode);
        }

        std::shared_ptr<Node> ScaledDequantize(std::shared_ptr<Node> input,
                                               std::shared_ptr<Node> min,
                                               std::shared_ptr<Node> max,
                                               const ngraph::element::Type& real_type,
                                               const ngraph::AxisSet& axes)
        {
            if (min->get_element_type() != real_type)
            {
                min = make_shared<op::Convert>(min, real_type);
            }

            if (max->get_element_type() != real_type)
            {
                max = make_shared<op::Convert>(max, real_type);
            }

            auto shape = min->get_shape();
            if (shape != max->get_shape())
            {
                throw ngraph_error("ScaledDequantize: min and max must have same shape");
            }

            auto quant_type = input->get_element_type();
            auto zero = op::Constant::create(quant_type, shape, {0});
            auto scale = builder::quantization_util::get_quantization_scale(min, max, quant_type);
            return make_shared<op::Dequantize>(input, scale, zero, real_type, axes);
        }

        std::shared_ptr<Node> ScaledQuantizedAvgPool(const std::shared_ptr<Node>& arg,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     bool include_padding_in_avg_computation,
                                                     const std::shared_ptr<Node> min,
                                                     const std::shared_ptr<Node> max)
        {
            return make_shared<op::QuantizedAvgPool>(arg,
                                                     window_shape,
                                                     window_movement_strides,
                                                     padding_below,
                                                     padding_above,
                                                     include_padding_in_avg_computation);
        }

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
                                           const bool with_relu)
        {
            auto requantization_scale = builder::quantization_util::get_scale(min_input,
                                                                              max_input,
                                                                              min_filter,
                                                                              max_filter,
                                                                              min_freezed_output,
                                                                              max_freezed_output);

            return make_shared<op::QuantizedConvolutionBias>(data_batch,
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
                                           const std::shared_ptr<Node> max_freezed_output)
        {
            auto requantization_scale = builder::quantization_util::get_scale(min_input,
                                                                              max_input,
                                                                              min_filter,
                                                                              max_filter,
                                                                              min_freezed_output,
                                                                              max_freezed_output);

            return make_shared<op::QuantizedConvolutionRelu>(data_batch,
                                                             filters,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale);
        }

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
                                       const std::shared_ptr<Node> max_freezed_output)
        {
            auto requantization_scale = builder::quantization_util::get_scale(min_input,
                                                                              max_input,
                                                                              min_filter,
                                                                              max_filter,
                                                                              min_freezed_output,
                                                                              max_freezed_output);

            return make_shared<op::QuantizedConvolution>(data_batch,
                                                         filters,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         requantization_scale);
        }

        std::shared_ptr<Node> ScaledQuantizedMaxPool(const std::shared_ptr<Node>& arg,
                                                     const Shape& window_shape,
                                                     const Strides& window_movement_strides,
                                                     const Shape& padding_below,
                                                     const Shape& padding_above,
                                                     const std::shared_ptr<Node> min,
                                                     const std::shared_ptr<Node> max)
        {
            return make_shared<op::QuantizedMaxPool>(
                arg, window_shape, window_movement_strides, padding_below, padding_above);
        }
    }
}
