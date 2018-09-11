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

#pragma once

#include <limits>
#include "ngraph/op/constant.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace quantization_util
            {
                template <class T1, class T2, class T3>
                void QuantizationRangeForMultiplication(
                    float min_a, float max_a, float min_b, float max_b, float* min_c, float* max_c)
                {
                    float a_one_quant_level = (max_a - min_a) / (std::numeric_limits<T1>::max() -
                                                                 std::numeric_limits<T1>::min());
                    float b_one_quant_level = (max_b - min_b) / (std::numeric_limits<T2>::max() -
                                                                 std::numeric_limits<T2>::min());
                    float c_one_quant_level = a_one_quant_level * b_one_quant_level;
                    *min_c = c_one_quant_level * std::numeric_limits<T3>::min();
                    *max_c = c_one_quant_level * std::numeric_limits<T3>::max();
                }

                const float get_scale(const ngraph::Node* node);
            }
        }
    }
}
