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

#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void avg_pool(void* arg,
                              void* out,
                              const Shape& arg_shape,
                              const Shape& out_shape,
                              const Shape& window_shape,
                              const Strides& window_movement_strides,
                              const Shape& padding_below,
                              const Shape& padding_above,
                              bool include_padding_in_avg_computation)
                {
                    reference::avg_pool<ElementType>(static_cast<const ElementType*>(arg),
                                                     static_cast<ElementType*>(out),
                                                     arg_shape,
                                                     out_shape,
                                                     window_shape,
                                                     window_movement_strides,
                                                     padding_below,
                                                     padding_above,
                                                     include_padding_in_avg_computation);
                }

                template <typename ElementType>
                void avg_pool_backprop(void* delta,
                                       void* out,
                                       const Shape& delta_shape,
                                       const Shape& out_shape,
                                       const Shape& window_shape,
                                       const Strides& window_movement_strides,
                                       const Shape& padding_below,
                                       const Shape& padding_above,
                                       bool include_padding_in_avg_computation)
                {
                    reference::avg_pool_backprop<ElementType>(
                        static_cast<const ElementType*>(delta),
                        static_cast<ElementType*>(out),
                        delta_shape,
                        out_shape,
                        window_shape,
                        window_movement_strides,
                        padding_below,
                        padding_above,
                        include_padding_in_avg_computation);
                }
            }
        }
    }
}
