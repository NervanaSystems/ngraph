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

#include "ngraph/runtime/reference/batch_norm.hpp"
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
                void batch_norm_training(double eps,
                                         const void* arg0,
                                         const void* arg1,
                                         const void* arg2,
                                         void* out0,
                                         void* out1,
                                         void* out2,
                                         const Shape& arg2_shape)
                {
                    reference::batch_norm_training(eps,
                                                   static_cast<const ElementType*>(arg0),
                                                   static_cast<const ElementType*>(arg1),
                                                   static_cast<const ElementType*>(arg2),
                                                   static_cast<ElementType*>(out0),
                                                   static_cast<ElementType*>(out1),
                                                   static_cast<ElementType*>(out2),
                                                   arg2_shape);
                }

                template <typename ElementType>
                void batch_norm_inference(double eps,
                                          const void* arg0,
                                          const void* arg1,
                                          const void* arg2,
                                          const void* arg3,
                                          const void* arg4,
                                          void* out0,
                                          const Shape& arg2_shape)
                {
                    reference::batch_norm_inference(eps,
                                                    static_cast<const ElementType*>(arg0),
                                                    static_cast<const ElementType*>(arg1),
                                                    static_cast<const ElementType*>(arg2),
                                                    static_cast<const ElementType*>(arg3),
                                                    static_cast<const ElementType*>(arg4),
                                                    static_cast<ElementType*>(out0),
                                                    arg2_shape);
                }

                template <typename ElementType>
                void batch_norm_backprop(double eps,
                                         const void* arg0,
                                         const void* arg1,
                                         const void* arg2,
                                         const void* arg3,
                                         const void* arg4,
                                         const void* arg5,
                                         void* out0,
                                         void* out1,
                                         void* out2,
                                         const Shape& arg2_shape)
                {
                    reference::batch_norm_backprop(eps,
                                                   static_cast<const ElementType*>(arg0),
                                                   static_cast<const ElementType*>(arg1),
                                                   static_cast<const ElementType*>(arg2),
                                                   static_cast<const ElementType*>(arg3),
                                                   static_cast<const ElementType*>(arg4),
                                                   static_cast<const ElementType*>(arg5),
                                                   static_cast<ElementType*>(out0),
                                                   static_cast<ElementType*>(out1),
                                                   static_cast<ElementType*>(out2),
                                                   arg2_shape);
                }
            }
        }
    }
}
