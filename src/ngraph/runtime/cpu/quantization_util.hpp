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

#include <unsupported/Eigen/CXX11/Tensor>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace quantization_util
            {
                template <class T>
                float FloatForOneQuantizedLevel(float range_min, float range_max)
                {
                    const int64 highest = static_cast<int64>(Eigen::NumTraits<T>::highest());
                    const int64 lowest = static_cast<int64>(Eigen::NumTraits<T>::lowest());
                    const float float_for_one_quantized_level =
                        (range_max - range_min) / (highest - lowest);
                    return float_for_one_quantized_level;
                }

                template <class T1, class T2, class T3>
                void QuantizationRangeForMultiplication(
                    float min_a, float max_a, float min_b, float max_b, float* min_c, float* max_c)
                {
                    const float a_float_for_one_quant_level =
                        FloatForOneQuantizedLevel<T1>(min_a, max_a);
                    const float b_float_for_one_quant_level =
                        FloatForOneQuantizedLevel<T2>(min_b, max_b);

                    const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
                    const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
                    const float c_float_for_one_quant_level =
                        a_float_for_one_quant_level * b_float_for_one_quant_level;

                    *min_c = c_float_for_one_quant_level * c_lowest;
                    *max_c = c_float_for_one_quant_level * c_highest;
                }
            }
        }
    }
}
