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
                auto input_min = min_input_const_op->get_vector<float>();
                auto input_max = max_input_const_op->get_vector<float>();
                auto filter_min = min_filter_const_op->get_vector<float>();
                auto filter_max = max_filter_const_op->get_vector<float>();
                auto output_min = min_freezed_output_const_op->get_vector<float>();
                auto output_max = max_freezed_output_const_op->get_vector<float>();

                float min_out_value;
                float max_out_value;
                quantization_range_for_multiplication<uint8_t, int8_t, int32_t>(input_min[0],
                                                                                input_max[0],
                                                                                filter_min[0],
                                                                                filter_max[0],
                                                                                &min_out_value,
                                                                                &max_out_value);
                const float max_abs32 = std::max(std::abs(min_out_value), std::abs(max_out_value));
                const float max_abs8 = std::max(std::abs(output_min[0]), std::abs(output_max[0]));
                // Output is signed int.
                // s32 = f32 * std::pow(2, 31)/ max_abs32;
                // s8 = f32 * std::pow(2, 7)/ max_abs8;
                // s8 = s32 * std::pow(2, -24) * max_abs32 / max_abs8;
                const float scale = static_cast<float>(
                    (std::pow(2, -24) * static_cast<double>(max_abs32 / max_abs8)));
                return scale;
            }

            template <typename T>
            static inline T get_quantization_scale(const std::shared_ptr<Node> min_input,
                                                   const std::shared_ptr<Node> max_input,
                                                   const ngraph::element::Type& type,
                                                   bool bump_by_eps = false)
            {
                auto min_input_const_op =
                    std::dynamic_pointer_cast<ngraph::op::Constant>(min_input);
                auto max_input_const_op =
                    std::dynamic_pointer_cast<ngraph::op::Constant>(max_input);

                if (min_input_const_op == nullptr)
                {
                    throw ngraph_error("min input must be constant");
                }
                else if (max_input_const_op == nullptr)
                {
                    throw ngraph_error("max input must be constant");
                }

                auto input_min_range = min_input_const_op->get_vector<T>();
                auto input_max_range = max_input_const_op->get_vector<T>();

                T min_range = std::numeric_limits<T>::min();
                T max_range = std::numeric_limits<T>::max();
                if (bump_by_eps)
                {
                    // If input_min_range and input_max_range are close,
                    // introduce a slightly larger delta between them.
                    min_range = std::min(static_cast<T>(0.0f), input_min_range[0]);
                    const T epsilon = std::max(static_cast<T>(1.0f),
                                               static_cast<T>(std::max(fabs(input_min_range[0]),
                                                                       fabs(input_max_range[0])))) /
                                      static_cast<T>(100.0f);
                    max_range = std::max(input_max_range[0], min_range + epsilon);
                    max_range = std::max(static_cast<T>(0.0f), max_range);
                    // end code copied and pasted from
                    // github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantize_op.cc
                }
                else
                {
                    min_range = input_min_range[0];
                    max_range = input_max_range[0];
                }

                const T max_abs = std::max(std::abs(min_range), std::abs(max_range));
                const T bitwidth = type.bitwidth();
                const T target_range = static_cast<T>(
                    (type.is_signed() ? std::pow(2, (bitwidth - 1)) : std::pow(2, bitwidth)) - 1);
                const T scale_factor = max_abs / target_range;
                return scale_factor;
            }
        }
    }
}
