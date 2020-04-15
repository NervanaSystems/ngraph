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

#include "ngraph/runtime/gpu/cublas_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
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
                                              size_t reduction_axes,
                                              const Node* node)
{
    std::stringstream ss;
    ss << "dot_op"
       << "_dtype_" << dtype.c_type_string() << "_reduction_axes_count_" << reduction_axes;
    std::string hash = ss.str() + "_i_" + join(arg0_shape, "_") + "_i_" + join(arg1_shape, "_");

    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> dot;
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& second = (arg0_shape.empty() ? arg1_shape : arg0_shape);
        size_t count = shape_size(second);

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

        primitive_index = this->m_primitive_emitter->register_primitive(dot, hash);
    }

    // case that can be treat as dot1d
    else if ((arg0_shape.size() == arg1_shape.size()) && (arg0_shape.size() == reduction_axes))
    {
        for (int i = 0; i < arg0_shape.size(); i++)
        {
            if (arg0_shape[i] != arg1_shape[i])
            {
                std::vector<std::string> arg_vec{"arg0", "arg1"};
                std::vector<Shape> shape_vec{arg0_shape, arg1_shape};
                throw std::invalid_argument(get_error_string(arg_vec, shape_vec, node));
            }
        }

        size_t count = shape_size(arg0_shape);
        dot.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            CUBLAS_SAFE_CALL(cublasSdot(*m_ctx->cublas_handle,
                                        count,
                                        static_cast<const float*>(inputs[0]),
                                        1,
                                        static_cast<const float*>(inputs[1]),
                                        1,
                                        static_cast<float*>(outputs[0])));

            debug_sync();
        }});

        primitive_index = this->m_primitive_emitter->register_primitive(dot, hash);
    }

    // matrix vector
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) && (reduction_axes == 1))
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

        primitive_index = this->m_primitive_emitter->register_primitive(dot, hash);
    }

    else
    {
        size_t axes_for_m_count = arg0_shape.size() - reduction_axes;
        size_t axes_for_n_count = arg1_shape.size() - reduction_axes;
        size_t axes_for_k_count = reduction_axes;
        size_t m = 1;
        size_t n = 1;
        size_t k = 1;

        // check if input and output size correct
        // check and calculate k for arg0 and arg1
        size_t arg0_k_idx = axes_for_m_count; // first axe in arg0 for k
        size_t arg1_k_idx = 0;                // first axe in arg1 for k

        for (size_t i = 0; i < axes_for_k_count; i++)
        {
            k *= arg0_shape[arg0_k_idx];
            if (arg0_shape[arg0_k_idx++] != arg1_shape[arg1_k_idx++])
            {
                std::vector<std::string> arg_vec{"arg0", "arg1"};
                std::vector<Shape> shape_vec{arg0_shape, arg1_shape};
                throw std::invalid_argument(get_error_string(arg_vec, shape_vec, node));
            }
        }
        // check and calculate m for arg0 and out
        size_t arg0_m_idx = 0; // first axe in arg0 for m
        size_t out_m_idx = 0;  // first axe in out for m
        for (size_t i = 0; i < axes_for_m_count; i++)
        {
            m *= arg0_shape[arg0_m_idx];
            if (arg0_shape[arg0_m_idx++] != out_shape[out_m_idx++])
            {
                std::vector<std::string> arg_vec{"arg0", "output"};
                std::vector<Shape> shape_vec{arg0_shape, out_shape};
                throw std::invalid_argument(get_error_string(arg_vec, shape_vec, node));
            }
        }
        // check and calculate n for arg1 and out
        size_t arg1_n_idx = axes_for_k_count; // first axe in arg1 for n
        size_t out_n_idx = axes_for_m_count;  // first axe in arg1 for n
        for (size_t i = 0; i < axes_for_n_count; i++)
        {
            n *= arg1_shape[arg1_n_idx];
            if (arg1_shape[arg1_n_idx++] != out_shape[out_n_idx++])
            {
                std::vector<std::string> arg_vec{"arg1", "output"};
                std::vector<Shape> shape_vec{arg1_shape, out_shape};
                throw std::invalid_argument(get_error_string(arg_vec, shape_vec, node));
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
            CUBLAS_SAFE_CALL(
                cublasSetPointerMode(*m_ctx->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

            debug_sync();
        }});
        primitive_index = this->m_primitive_emitter->register_primitive(dot, hash);
    }

    return primitive_index;
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

std::string runtime::gpu::CUBLASEmitter::get_error_string(std::vector<std::string>& arg_names,
                                                          std::vector<Shape>& shapes,
                                                          const Node* node)
{
    std::stringstream ss_err;
    ss_err << ngraph::join(arg_names) << " with " << ngraph::join(shapes)
           << " respectively, at Node " << node->get_name() << ", do not match for dot op";

    return ss_err.str();
}
