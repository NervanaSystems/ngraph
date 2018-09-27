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
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/runtime/gpu/cublas_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

runtime::gpu::CUBLASEmitter::CUBLASEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx)
    : m_primitive_emitter(emitter)
{
    m_ctx = ctx;
}

size_t runtime::gpu::CUBLASEmitter::build_dot(const element::Type& dtype,
                                              const Shape& arg0_shape,
                                              const Shape& arg1_shape,
                                              const Shape& out_shape,
                                              size_t reductionAxesCount)
{
    std::stringstream ss;
    ss << "dot_op"
       << "_dtype_" << dtype.c_type_string();
    std::string hash = ss.str() + "_i_" + join(arg0_shape, "_") + "_i_" + join(arg1_shape, "_") +
                       "_o_" + join(out_shape, "_");

    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> dot;
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& second = (arg0_shape.empty() ? arg1_shape : arg0_shape);
        int count = shape_size(second);

        size_t firstIndex = (arg0_shape.empty() ? 0 : 1);
        size_t secondIndex = (arg0_shape.empty() ? 1 : 0);

        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            CUBLAS_SAFE_CALL(cublasScopy(*m_ctx->cublas_handle,
                                         count,
                                         static_cast<const float*>(inputs[secondIndex]),
                                         1,
                                         static_cast<float*>(outputs[0]),
                                         1));
            CUBLAS_SAFE_CALL(cublasSscal(*m_ctx->cublas_handle,
                                         count,
                                         static_cast<const float*>(inputs[firstIndex]),
                                         static_cast<float*>(outputs[0]),
                                         1));
            debug_sync();
        }});

        // primitive_index = this->m_primitive_emitter->insert(std::move(dot));
        // m_primitive_emitter->cache(hash, primitive_index);
        return getPrimitiveIndex(dot, hash);
    }

    if (shape_size(arg0_shape) == 0 || shape_size(arg1_shape) == 0)
    {
        size_t elemSize = shape_size(out_shape) * dtype.size();
        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            runtime::gpu::cuda_memset(outputs[0], 0, elemSize);
            debug_sync();
        }});

        // primitive_index = this->m_primitive_emitter->insert(std::move(dot));
        // m_primitive_emitter->cache(hash, primitive_index);
        return getPrimitiveIndex(dot, hash);
    }

    if ()
    {

    
    }
     else if
     {

     }

    else
    {

    }
}

void runtime::gpu::CUBLASEmitter::sync()
{
    CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
    return;
}

void runtime::gpu::CUBLASEmitter::debug_sync()
{
#ifdef NGRAPH_DEBUG_ENABLE
    CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return;
}

size_t runtime::gpu::CUBLASEmitter::getPrimitiveIndex(std::unique_ptr<gpu::primitive>& dot,
                                                      std::string hash)
{
    size_t primitive_index = this->m_primitive_emitter->insert(std::move(dot));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}
