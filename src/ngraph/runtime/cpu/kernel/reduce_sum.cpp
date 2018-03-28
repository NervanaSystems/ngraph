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

#include "reduce_sum.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void reduce_sum_all_2d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape)
                {
                    reduce_sum_all<float, 2>(input, output, input_shape, output_shape);
                }

                void reduce_sum_2d_1rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes)
                {
                    reduce_sum<float, 2, 1>(input, output, input_shape, output_shape, reduction_axes);
                }

            }
        }
    }
}
