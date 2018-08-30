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

#include "reshape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void reshape_3d_3d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape)
                {
                    reshape<float, 3, 3>(
                        input, output, input_shape, input_axis_order, output_shape);
                }

                void reshape_4d_4d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape)
                {
                    reshape<float, 4, 4>(
                        input, output, input_shape, input_axis_order, output_shape);
                }
            }
        }
    }
}
