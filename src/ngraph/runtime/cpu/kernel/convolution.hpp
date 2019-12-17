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

#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename INPUT,
                          typename FILTER,
                          typename OUTPUT,
                          typename ACCUMULATION =
                              typename ngraph::runtime::reference::widen<OUTPUT>::type>
                void convolution(void* input0,
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
                                 void* input_scale,
                                 void* input_zero_point,
                                 void* filter_scale,
                                 void* filter_zero_point,
                                 void* output_scale,
                                 void* output_zero_point)
                {
                    reference::convolution<INPUT, FILTER, OUTPUT, ACCUMULATION>(
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
                        static_cast<const float*>(input_scale),
                        static_cast<const INPUT*>(input_zero_point),
                        static_cast<const float*>(filter_scale),
                        static_cast<const FILTER*>(filter_zero_point),
                        static_cast<const float*>(output_scale),
                        static_cast<const OUTPUT*>(output_zero_point));
                }

                template <typename ElementType>
                void convolution_backprop_filter(void* input0,
                                                 void* input1,
                                                 void* output,
                                                 const Shape& arg0_shape,
                                                 const Shape& arg1_shape,
                                                 const Shape& filter_shape,
                                                 const Strides& window_dilation_strides,
                                                 const Strides& window_movement_strides,
                                                 const CoordinateDiff& padding_below,
                                                 const CoordinateDiff& padding_above,
                                                 const Strides& data_dilation_strides)
                {
                    reference::convolution_backprop_filter<ElementType>(
                        static_cast<const ElementType*>(input0),
                        static_cast<const ElementType*>(input1),
                        static_cast<ElementType*>(output),
                        arg0_shape,
                        arg1_shape,
                        filter_shape,
                        window_dilation_strides,
                        window_movement_strides,
                        padding_below,
                        padding_above,
                        data_dilation_strides);
                }

                template <typename ElementType>
                void convolution_backprop_in(void* input0,
                                             void* input1,
                                             void* output,
                                             const Shape& arg0_shape,
                                             const Shape& arg1_shape,
                                             const Shape& in_shape,
                                             const Strides& window_movement_strides,
                                             const Strides& window_dilation_strides,
                                             const CoordinateDiff& padding_below,
                                             const CoordinateDiff& padding_above,
                                             const Strides& data_dilation_strides)
                {
                    reference::convolution_backprop_in<ElementType>(
                        static_cast<const ElementType*>(input0),
                        static_cast<const ElementType*>(input1),
                        static_cast<ElementType*>(output),
                        arg0_shape,
                        arg1_shape,
                        in_shape,
                        window_movement_strides,
                        window_dilation_strides,
                        padding_below,
                        padding_above,
                        data_dilation_strides);
                }
            } // namespace kernel
        }     // namespace cpu
    }         // namespace runtime
} // namespace ngraph
