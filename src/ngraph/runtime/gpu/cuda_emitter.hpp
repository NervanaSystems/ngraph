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

#include "ngraph/shape.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUPrimitiveEmitter;

            class CUDAEmitter
            {
                friend class GPUPrimitiveEmitter;
            public:
                size_t build_pad(const GPURuntimeContext* ctx,
                                 const std::array<std::string, 2>& dtypes,
                                 const ngraph::Shape& input_shape,
                                 const ngraph::Shape& output_shape,
                                 const ngraph::Shape& pad_below,
                                 const ngraph::Shape& pad_above,
                                 const ngraph::Shape& pad_interior);
            private:
                CUDAEmitter(GPUPrimitiveEmitter* emitter);

                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
