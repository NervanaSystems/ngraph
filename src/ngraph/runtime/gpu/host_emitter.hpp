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

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            struct GPURuntimeContext;
            class GPUPrimitiveEmitter;
            class HostEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                size_t build_memcpy(const cudaMemcpyKind& kind,
                                    const size_t& size,
                                    const size_t& dst = 0,
                                    const size_t& src = 0);
                size_t build_zero_out(const size_t& dst, const size_t& size, bool is_local = false);

            private:
                HostEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx);

                GPUPrimitiveEmitter* m_primitive_emitter;
                GPURuntimeContext* m_ctx;
            };
        }
    }
}
