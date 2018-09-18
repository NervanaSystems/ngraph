/*******************************************************************************
* Copyright 2018 Intel Corporation
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
                float get_scale(const ngraph::Node* node)
                {
                    auto qconvolution = static_cast<const ngraph::op::QuantizedConvolution*>(node);
                    float min_out_value;
                    float max_out_value;
                    quantization_range_for_multiplication<uint8_t, int8_t, int32_t>(
                        qconvolution->get_input_min(),
                        qconvolution->get_input_max(),
                        qconvolution->get_filter_min(),
                        qconvolution->get_filter_max(),
                        &min_out_value,
                        &max_out_value);
                    const float max_abs32 =
                        std::max(std::abs(min_out_value), std::abs(max_out_value));
                    const float max_abs8 =
                        std::max(std::abs(qconvolution->get_freezed_output_min()),
                                 std::abs(qconvolution->get_freezed_output_max()));
                    // Output is signed int.
                    // s32 = f32 * std::pow(2, 31)/ max_abs32;
                    // s8 = f32 * std::pow(2, 7)/ max_abs8;
                    // s8 = s32 * std::pow(2, -24) * max_abs32 / max_abs8;
                    const float scale = static_cast<float>(
                        (std::pow(2, -24) * static_cast<double>(max_abs32 / max_abs8)));
                    return scale;
                }
            }
        }
    }
}
