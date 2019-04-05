//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#pragma once

#include "ngraph/runtime/reference/quantized_convolution.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <class INPUT, class FILTER, class OUTPUT>
                void quant_convolution(void* input0,
                                       void* input1,
                                       void* output,
                                       const Shape& arg0_shape,
                                       const Shape& arg1_shape,
                                       const Shape& result_shape,
                                       const Strides& window_movement_strides,
                                       const Strides& window_dilation_strides,
                                       const CoordinateDiff& padding_below,
                                       const CoordinateDiff& padding_above,
                                       const Strides& data_dilation_strides,
                                       const float requant_scale)
                {
                    reference::quant_convolution<INPUT, FILTER, OUTPUT>(
                        static_cast<const INPUT*>(input0),
                        static_cast<const FILTER*>(input1),
                        static_cast<OUTPUT*>(output),
                        arg0_shape,
                        arg1_shape,
                        result_shape,
                        window_movement_strides,
                        window_dilation_strides,
                        padding_below,
                        padding_above,
                        data_dilation_strides,
                        requant_scale);
                }

            } // namespace kernel
        }     // namespace cpu
    }         // namespace runtime
} // namespace ngraph
