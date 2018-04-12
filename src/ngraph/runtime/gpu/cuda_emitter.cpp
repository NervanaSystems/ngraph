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
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/cuda_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

runtime::gpu::CUDAEmitter::CUDAEmitter(runtime::gpu::GPUPrimitiveEmitter* emitter)
    : m_primitive_emitter(emitter)
{
}

size_t runtime::gpu::CUDAEmitter::build_pad(const runtime::gpu::GPURuntimeContext* ctx,
                                            const std::array<std::string, 2>& dtypes,
                                            const Shape& input_shape,
                                            const Shape& output_shape,
                                            const Shape& padding_below,
                                            const Shape& padding_above,
                                            const Shape& padding_interior)
{
    if (padding_interior.size())
    {
        throw std::runtime_error("Interior padding is not yet supported in the GPU transformer.");
    }
    std::string hash = "pad_i" + join(input_shape, "_") + "_pb" + join(padding_below, "_") + "_pa" +
                       join(padding_above, "_") + "_pi" + join(padding_interior, "_");

    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list
    // Once all previously implemented cuda kernels are refactored to use the
    // CUDAEmitter/GPUPrimittiveEmitter interface, only one map (from hash to primitive index)
    // will be required.

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // if the kernel has not been compiled, build it
    auto compiled_kernel = ctx->compiled_kernel_pool->get(hash);
    auto nthreads = shape_size(input_shape);
    if (compiled_kernel == nullptr)
    {
        // normalize pad dimensions to shape dimensions
        Shape pad_below(input_shape.size(), 0);
        Shape pad_above(input_shape.size(), 0);
        if (padding_below.size() != input_shape.size())
        {
            for (int64_t i = padding_below.size() - 1; i >= 0; i--)
            {
                pad_below[i + input_shape.size() - padding_below.size()] = padding_below[i];
                pad_above[i + input_shape.size() - padding_above.size()] = padding_above[i];
            }
        }
        else
        {
            pad_below = padding_below;
            pad_above = padding_above;
        }

        auto input_strides = row_major_strides(input_shape);
        auto output_strides = row_major_strides(output_shape);

        int offset = 0;
        for (size_t i = 0; i < output_strides.size(); i++)
        {
            offset += (output_strides[i] * pad_below[i]);
        }

        codegen::CodeWriter writer;
        writer << "extern \"C\" __global__ void cuda_" << hash << "(" << dtypes[0] << "* in, "
               << dtypes[1] << "* out)\n";
        writer.block_begin();
        {
            writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
            writer << "if (tid < " << nthreads << ")\n";
            writer.block_begin();
            {
                writer << "size_t idx = ";
                writer << offset << " + tid % " << input_shape.back();
                int64_t last = input_strides.size() - 1;
                for (int64_t i = last - 1; i > 0; i--)
                {
                    writer << " + ((tid / " << input_strides[i] << ") % " << input_shape[i + 1]
                           << ") * " << output_strides[i];
                }
                writer << ";\n";
                writer << "out[idx] = in[tid];\n";
            }
            writer.block_end();
        }
        writer.block_end();
        compiled_kernel = ctx->compiled_kernel_pool->set(hash, writer.get_code());
    }

    auto pad = new gpu::primitive{[=](void** inputs, void** outputs) {

        void* args_list[] = {&inputs[0], &outputs[0]};
        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      static_cast<unsigned int>(nthreads),
                                      1,
                                      1, // grid dim
                                      1,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list,
                                      0));  // arguments
        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    }};

    primitive_index = this->m_primitive_emitter->insert(pad);
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

void print_tensor_from_gpu(codegen::CodeWriter& writer,
                           const std::string& tensor_name,
                           const Shape& shape)
{
    auto strides = row_major_strides(shape);
    writer << "__syncthreads();\n";
    writer << "if (tid==0)\n";
    writer.block_begin();
    {
        std::string element = tensor_name + "[i]";
        writer << "for (int i=0; i<" << shape_size(shape) << "; i++)\n";
        writer.block_begin();
        {
            for (int64_t i = strides.size() - 1; i >= 0; i--)
            {
                writer << "if (i % " << strides[i] << " == 0)\n";
                writer.block_begin();
                {
                    writer << "printf(\"";
                    for (int64_t j = 0; j < strides.size() - 1 - i; j++)
                    {
                        writer << "\\n";
                    }
                    writer << "\");\n";
                }
                writer.block_end();
            }
            writer << "printf(\"%4.2f \", " << element << ");\n";
        }
        writer.block_end();
        writer << "printf(\"\\n\");\n";
    }
    writer.block_end();
}
