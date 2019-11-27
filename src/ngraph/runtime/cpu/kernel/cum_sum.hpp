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

#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename InputElementType, typename AxisElementType>
                void reference_cumsum(void* input_tensor,
                                      void* axis_tensor,
                                      void* out,
                                      const Shape& tensor_shape,
                                      const bool exclusive,
                                      const bool reverse)
                {
                    reference::cumsum<InputElementType, AxisElementType>(
                        static_cast<const InputElementType*>(input_tensor),
                        static_cast<const AxisElementType*>(axis_tensor),
                        static_cast<InputElementType*>(out),
                        tensor_shape,
                        exclusive,
                        reverse);
                }
            }
        }
    }
}
