/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "quantization_util.hpp"
#include "ngraph/runtime/cpu/op/quantized_conv.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace quantization_util
            {
                const float get_output_scale(const ngraph::Node* node)
                {
                    auto qconvolution = static_cast<const ngraph::op::QuantizedConvolution*>(node);
                    auto min_input_const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        qconvolution->get_argument(2));
                    auto max_input_const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        qconvolution->get_argument(3));
                    auto min_filter_const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        qconvolution->get_argument(4));
                    auto max_filter_const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        qconvolution->get_argument(5));
                    auto min_freezed_output_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(
                            qconvolution->get_argument(6));
                    auto max_freezed_output_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(
                            qconvolution->get_argument(7));

                    float min_input =
                        *(static_cast<float const*>(min_input_const_op->get_data_ptr()));
                    float max_input =
                        *(static_cast<float const*>(max_input_const_op->get_data_ptr()));
                    float min_filter =
                        *(static_cast<float const*>(min_filter_const_op->get_data_ptr()));
                    float max_filter =
                        *(static_cast<float const*>(max_filter_const_op->get_data_ptr()));
                    float min_output =
                        *(static_cast<float const*>(min_freezed_output_const_op->get_data_ptr()));
                    float max_output =
                        *(static_cast<float const*>(max_freezed_output_const_op->get_data_ptr()));

                    float min_out_value;
                    float max_out_value;
                    QuantizationRangeForMultiplication<uint8_t, int8_t, int32_t>(min_input,
                                                                                 max_input,
                                                                                 min_filter,
                                                                                 max_filter,
                                                                                 &min_out_value,
                                                                                 &max_out_value);
                    const float max_abs32 =
                        std::max(std::abs(min_out_value), std::abs(max_out_value));
                    const float max_abs8 = std::max(std::abs(min_output), std::abs(max_output));
                    // Output is signed int.
                    // s32 = f32 * std::pow(2, 31)/ max_abs32;
                    // s8 = f32 * std::pow(2, 7)/ max_abs8;
                    // s8 = s32 * std::pow(2, -24) * max_abs32 / max_abs8;
                    const float scale = std::pow(2, -24) * max_abs32 / max_abs8;

                    return scale;
                }
            }
        }
    }
}
