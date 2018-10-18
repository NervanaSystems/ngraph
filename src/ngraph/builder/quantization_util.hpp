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
//*******************************************************************************
//  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//==============================================================================

#pragma once

#include <limits>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization_util
        {
            template <class T1, class T2, class T3>
            void quantization_range_for_multiplication(
                float min_a, float max_a, float min_b, float max_b, float* min_c, float* max_c)
            {
                // begin code copied and pasted (and modified) from
                // github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantization_utils.h
                float a_one_quant_level = (max_a - min_a) / (std::numeric_limits<T1>::max() -
                                                             std::numeric_limits<T1>::min());
                float b_one_quant_level = (max_b - min_b) / (std::numeric_limits<T2>::max() -
                                                             std::numeric_limits<T2>::min());
                float c_one_quant_level = a_one_quant_level * b_one_quant_level;
                *min_c = c_one_quant_level * std::numeric_limits<T3>::min();
                *max_c = c_one_quant_level * std::numeric_limits<T3>::max();
                // end code copied and pasted (and modified) from
                // github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantization_utils.h
            }

            float get_scale(const std::shared_ptr<Node> min_input,
                            const std::shared_ptr<Node> max_input,
                            const std::shared_ptr<Node> min_filter,
                            const std::shared_ptr<Node> max_filter,
                            const std::shared_ptr<Node> min_freezed_output,
                            const std::shared_ptr<Node> max_freezed_output)
            {
                auto min_input_const_op = std::static_pointer_cast<ngraph::op::Constant>(min_input);
                auto max_input_const_op = std::static_pointer_cast<ngraph::op::Constant>(max_input);
                auto min_filter_const_op =
                    std::static_pointer_cast<ngraph::op::Constant>(min_filter);
                auto max_filter_const_op =
                    std::static_pointer_cast<ngraph::op::Constant>(max_filter);
                auto min_freezed_output_const_op =
                    std::static_pointer_cast<ngraph::op::Constant>(min_freezed_output);
                auto max_freezed_output_const_op =
                    std::static_pointer_cast<ngraph::op::Constant>(max_freezed_output);
                float input_min = *(static_cast<float const*>(min_input_const_op->get_data_ptr()));
                float input_max = *(static_cast<float const*>(max_input_const_op->get_data_ptr()));
                float filter_min =
                    *(static_cast<float const*>(min_filter_const_op->get_data_ptr()));
                float filter_max =
                    *(static_cast<float const*>(max_filter_const_op->get_data_ptr()));
                float output_min =
                    *(static_cast<float const*>(min_freezed_output_const_op->get_data_ptr()));
                float output_max =
                    *(static_cast<float const*>(max_freezed_output_const_op->get_data_ptr()));

                float min_out_value;
                float max_out_value;
                quantization_range_for_multiplication<uint8_t, int8_t, int32_t>(
                    input_min, input_max, filter_min, filter_max, &min_out_value, &max_out_value);
                const float max_abs32 = std::max(std::abs(min_out_value), std::abs(max_out_value));
                const float max_abs8 = std::max(std::abs(output_min), std::abs(output_max));
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
