//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/host_emitter.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

runtime::gpu::HostEmitter::HostEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx)
    : m_primitive_emitter(emitter)
    , m_ctx(ctx)
{
}

size_t runtime::gpu::HostEmitter::build_memcpy(const cudaMemcpyKind& kind,
                                               size_t size,
                                               size_t dst,
                                               size_t src)
{
    std::stringstream ss;
    ss << "memcpy" << kind << "_dst" << dst << "_src" << src << "_sz" << size;
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> launch_kernel(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            CUDA_RT_SAFE_CALL(cudaMemcpy(outputs[dst], inputs[src], size, kind));
        }});

    return this->m_primitive_emitter->register_primitive(launch_kernel, hash);
}

size_t runtime::gpu::HostEmitter::build_zero_out(size_t dst, size_t size, bool is_local)
{
    std::stringstream ss;
    ss << "zero"
       << "_dst" << dst << "_sz" << size << "_local" << is_local;
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> launch_kernel;
    if (is_local)
    {
        launch_kernel.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* tensor = gpu::invoke_memory_primitive(m_ctx, dst);
            CUDA_RT_SAFE_CALL(cudaMemset(tensor, 0, size));
        }});
    }
    else
    {
        launch_kernel.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            CUDA_RT_SAFE_CALL(cudaMemset(outputs[dst], 0, size));
        }});
    }

    return this->m_primitive_emitter->register_primitive(launch_kernel, hash);
}
