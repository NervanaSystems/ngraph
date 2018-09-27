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
#include <stdexcept>

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
using namespace std;

runtime::gpu::CUBLASEmitter::CUBLASEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx)
    : m_primitive_emitter(emitter)
{
    m_ctx = ctx;
}

size_t runtime::gpu::CUBLASEmitter::build_dot(const element::Type& dtype,
                                              const Shape& arg0_shape,
                                              const Shape& arg1_shape,
                                              const Shape& out_shape,
                                              size_t reduction_axes)
{
    std::stringstream ss;
    ss << "dot_op"
       << "_dtype_" << dtype.c_type_string();
    std::string hash = ss.str() + "_i_" + join(arg0_shape, "_") + "_i_" + join(arg1_shape, "_") +
                       "_o_" + join(out_shape, "_") + "_reduction_axes_count_" + reduction_axes;

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

        return getPrimitiveIndex(dot, hash);
    }

    if (shape_size(arg0_shape) == 0 || shape_size(arg1_shape) == 0)
    {
        size_t elemSize = shape_size(out_shape) * dtype.size();
        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            runtime::gpu::cuda_memset(outputs[0], 0, elemSize);
            debug_sync();
        }});

        return getPrimitiveIndex(dot, hash);
    }
    // case that can be treat as dot1d
    if ((arg0_shape.size() == arg1_shape.size()) && (arg0_shape.size() == reduction_axes))
    {
        for (int i = 0; i < arg0_shape.size(); i++)
        {
            if (arg0_shape[i] != arg1_shape[i])
            {
                throw invalid_argument("arg0 and arg1 shape does not match for dot.");
            }
        }

        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            CUBLAS_SAFE_CALL(cublasSdot(*m_ctx->cublas_handle,
                                        shape_size(arg0_shape),
                                        static_cast<const float*>(inputs[0]),
                                        1,
                                        static_cast<const float*>(inputs[1]),
                                        1,
                                        static_cast<float*>(outputs[0])));

            debug_sync();
        }});

        return getPrimitiveIndex(dot, hash);
    }

    // matrix vector
    if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) && (reduction_axes == 1))
    {
        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            const float alpha = 1.0;
            const float beta = 0;

            CUBLAS_SAFE_CALL(cublasSetPointerMode(*m_ctx->cublas_handle, CUBLAS_POINTER_MODE_HOST));
            CUBLAS_SAFE_CALL(cublasSgemv(*m_ctx->cublas_handle,
                                         CUBLAS_OP_T,
                                         arg0_shape[1],
                                         arg0_shape[0],
                                         &alpha,
                                         static_cast<const float*>(inputs[0]),
                                         arg0_shape[1],
                                         static_cast<const float*>(inputs[1]),
                                         1,
                                         &beta,
                                         static_cast<float*>(outputs[0]),
                                         1));
            CUBLAS_SAFE_CALL(
                cublasSetPointerMode(*m_ctx->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

            debug_sync();
        }});

        return getPrimitiveIndex(dot, hash);
    }

    size_t num_of_axes_for_m = arg0_shape.size() - reduction_axes;
    size_t num_of_axes_for_n = arg1_shape.size() - reduction_axes;
    size_t num_of_axes_for_k = reduction_axes;
    size_t m = 1;
    size_t n = 1;
    size_t k = 1;

    // check if input and output size correct
    // check and calculate k for arg0 and arg1
    size_t arg0_k_idx = num_of_axes_for_m; // first axe in arg0 for k
    size_t arg1_k_idx = 0;                 // first axe in arg1 for k
    for (size_t i = 0; i < num_of_axes_for_k; i++)
    {
        k *= arg0_shape[arg0_k_idx];
        if (arg0_shape[arg0_k_idx++] != arg1_shape[arg1_k_idx++])
        {
            throw invalid_argument("arg0 and arg1 shape does not match for dot.");
        }
    }
    // check and calculate m for arg0 and out
    size_t arg0_m_idx = 0; // first axe in arg0 for m
    size_t out_m_idx = 0;  // first axe in out for m
    for (size_t i = 0; i < num_of_axes_for_m; i++)
    {
        m *= arg0_shape[arg0_m_idx];
        if (arg0_shape[arg0_m_idx++] != out_shape[out_m_idx++])
        {
            throw invalid_argument("arg0 and output shape does not match for dot.");
        }
    }
    // check and calculate n for arg1 and out
    size_t arg1_n_idx = num_of_axes_for_k; // first axe in arg1 for n
    size_t out_n_idx = num_of_axes_for_m;  // first axe in arg1 for n
    for (size_t i = 0; i < num_of_axes_for_n; i++)
    {
        n *= arg1_shape[arg1_n_idx];
        if (arg1_shape[arg1_n_idx++] != out_shape[out_n_idx++])
        {
            throw invalid_argument("arg1 and output shape does not match for dot.");
        }
    }

    dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
        const float alpha = 1.0;
        const float beta = 0;

        CUBLAS_SAFE_CALL(cublasSetPointerMode(*m_ctx->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_SAFE_CALL(cublasSgemm(*m_ctx->cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     n,
                                     m,
                                     k,
                                     &alpha,
                                     static_cast<const float*>(inputs[1]),
                                     n,
                                     static_cast<const float*>(inputs[0]),
                                     k,
                                     &beta,
                                     static_cast<float*>(outputs[0]),
                                     n));
        CUBLAS_SAFE_CALL(cublasSetPointerMode(*m_ctx->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

        debug_sync();
    }});

    return getPrimitiveIndex(dot, hash);
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
