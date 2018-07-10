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

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void broadcast(void* input0,
                               void* output,
                               const Shape& arg0_shape,
                               const Shape& result_shape,
                               const AxisSet& broadcast_axes)
                {
                    reference::broadcast<ElementType>(static_cast<const ElementType*>(input0),
                                                      static_cast<ElementType*>(output),
                                                      arg0_shape,
                                                      result_shape,
                                                      broadcast_axes);
                }
            }
        }
    }
}
