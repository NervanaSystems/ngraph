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
#include <ostream>
#include <string>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/cuda_emitter.hpp"
#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

struct pooling_op_shape
{
    int N;
    int C;
    int D;
    int H;
    int W;
    int K;
    int M;
    int P;
    int Q;
    int J;
    int T;
    int R;
    int S;
    int STRIDE_D;
    int STRIDE_H;
    int STRIDE_W;
    int PAD_D;
    int PAD_H;
    int PAD_W;
};

std::ostream& operator<<(std::ostream& os, pooling_op_shape& shape)
{
    return os << shape.N << "_" << shape.C << "_" << shape.D << "_" << shape.H << "_" << shape.W
              << "_" << shape.K << "_" << shape.M << "_" << shape.P << "_" << shape.Q << "_"
              << shape.J << "_" << shape.T << "_" << shape.R << "_" << shape.S << "_"
              << shape.STRIDE_D << "_" << shape.STRIDE_H << "_" << shape.STRIDE_W << "_"
              << shape.PAD_D << "_" << shape.PAD_H << "_" << shape.PAD_W;
}

runtime::gpu::CUDAEmitter::CUDAEmitter(runtime::gpu::GPUPrimitiveEmitter* emitter,
                                       runtime::gpu::GPURuntimeContext* ctx)
    : m_primitive_emitter(emitter)
{
    m_ctx = ctx;
}

size_t runtime::gpu::CUDAEmitter::build_concat(const std::vector<std::string>& dtypes,
                                               std::vector<NVShape> input_shapes,
                                               size_t concat_axis,
                                               NVShape output_shape)
{
    std::stringstream kernel_name;
    size_t input_size = input_shapes.size();
    kernel_name << "concat_" << join(dtypes, "_") << "_r_" << input_size;

    std::stringstream hash;
    hash << kernel_name.str() << "_o_" << join(output_shape, "_") << "_a_" << concat_axis;
    for (size_t i = 0; i < input_size; i++)
    {
        hash << "_i_" << join(input_shapes[i], "_");
    }
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash.str());
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_concat_op(writer, kernel_name.str(), dtypes, input_shapes.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    std::vector<uint32_t> block_strides(input_size, 1);
    uint32_t block_size = 0;
    for (size_t i = 0; i < input_size; i++)
    {
        auto arg_rank = input_shapes[i].size();
        for (size_t j = concat_axis; j < arg_rank; j++)
        {
            block_strides[i] *= input_shapes[i][j];
        }
        block_size += block_strides[i];
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t idx_block_strides =
        allocator.reserve_argspace(block_strides.data(), block_strides.size() * sizeof(uint32_t));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        void* param_block_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_block_strides);
        std::vector<void*> args_list;
        for (size_t i = 0; i < input_size; i++)
        {
            args_list.push_back(&inputs[i]);
        }
        args_list.push_back(&outputs[0]);
        args_list.push_back(&param_block_strides);
        args_list.push_back(&block_size);
        args_list.push_back(&nthreads);

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      aligned_grid_size_x,
                                      1,
                                      1, // grid dim
                                      block_size_x,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash.str(), primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_onehot(const std::array<std::string, 2>& dtypes,
                                               NVShape input_shape,
                                               NVShape output_shape,
                                               size_t one_hot_axis)
{
    std::stringstream kernel_name;
    kernel_name << "onehot_" << join(dtypes, "_");

    std::string hash = kernel_name.str() + "_i_" + join(input_shape, "_") + "_o_" +
                       join(output_shape, "_") + std::to_string(one_hot_axis);
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_onehot_op(writer, kernel_name.str(), dtypes);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    uint32_t repeat_times = static_cast<uint32_t>(output_shape[one_hot_axis]);
    uint32_t repeat_size = 1;
    for (size_t i = one_hot_axis + 1; i < output_shape.size(); i++)
    {
        repeat_size *= output_shape[i];
    }

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            std::vector<void*> args_list{
                &inputs[0], &outputs[0], &repeat_size, &repeat_times, &nthreads};
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list.data(),
                                          0)); // arguments
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_reverse(const std::array<std::string, 2>& dtypes,
                                                NVShape input_shape,
                                                std::vector<uint32_t> reverse_axes)
{
    uint32_t rank = static_cast<uint32_t>(input_shape.size());
    std::stringstream kernel_name;
    kernel_name << "reverse_" << join(dtypes, "_");

    std::string hash = kernel_name.str() + "_i_" + join(input_shape, "_") + "_axes_" +
                       join(reverse_axes, "_") + std::to_string(rank);
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_reverse_op(writer, kernel_name.str(), dtypes);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t idx_input_shape =
        allocator.reserve_argspace(input_shape.data(), input_shape.size() * sizeof(uint32_t));
    size_t idx_reverse_axes =
        allocator.reserve_argspace(reverse_axes.data(), reverse_axes.size() * sizeof(uint32_t));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        void* param_input_shape = runtime::gpu::invoke_memory_primitive(m_ctx, idx_input_shape);
        void* param_reverse_axes = runtime::gpu::invoke_memory_primitive(m_ctx, idx_reverse_axes);
        std::vector<void*> args_list{
            &inputs[0], &outputs[0], &param_input_shape, &param_reverse_axes, &rank, &nthreads};

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      aligned_grid_size_x,
                                      1,
                                      1, // grid dim
                                      block_size_x,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_pad(const std::array<std::string, 2>& dtypes,
                                            NVShape input_shape,
                                            NVShape output_shape,
                                            NVShape padding_below,
                                            NVShape padding_above,
                                            NVShape padding_interior,
                                            const std::string& pad_value)
{
    // Need to check: are there models in which some tensors will have different types? if so, this
    // hash needs to include the tensor types.
    std::string val_hash = (pad_value == "") ? "0" : "1";
    std::string hash = "pad_i" + join(input_shape, "_") + "_pb" + join(padding_below, "_") + "_pa" +
                       join(padding_above, "_") + "_pi" + join(padding_interior, "_") + "_pv" +
                       val_hash;

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

    size_t nthreads = shape_size(output_shape);
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(hash);
    if (compiled_kernel == nullptr)
    {
        // normalize pad dimensions to shape dimensions
        NVShape pad_below(input_shape.size(), 0);
        NVShape pad_above(input_shape.size(), 0);
        NVShape pad_interior(input_shape.size(), 0);

        // if padding_interior is not zero length, it
        // is from op::Pad for which padding_below will
        // always be equal in size to padding_above
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
            pad_interior = padding_interior;
        }

        NVShape input_strides = row_major_strides(input_shape);
        NVShape output_strides = row_major_strides(output_shape);

        int offset = 0;
        for (size_t i = 0; i < output_strides.size(); i++)
        {
            offset += (output_strides[i] * pad_below[i]);
        }

        codegen::CodeWriter writer;
        writer << "extern \"C\" __global__ void cuda_" << hash << "(";

        // if the pad value is static, a runtime argument isn't necessary
        if (pad_value == "")
        {
            writer << dtypes[0] << "* val, ";
        }
        writer << dtypes[0] << "* in, " << dtypes[1] << "* out)\n";
        writer.block_begin();
        {
            writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";

            // fill kernel
            writer << "if (tid < " << nthreads << ")\n";
            writer.block_begin();
            {
                if (pad_value == "")
                {
                    writer << "out[tid] = *val;\n";
                }
                else
                {
                    writer << "out[tid] = " << pad_value << ";\n";
                }
            }
            writer.block_end();

            // pad re-index kernel
            writer << "if (tid < " << shape_size(input_shape) << ")\n";
            writer.block_begin();
            {
                writer << "size_t idx = ";
                writer << offset << " + (tid % " << input_shape.back() << ") * "
                       << 1 + pad_interior.back();
                int64_t last = input_strides.size() - 1;
                for (int64_t i = last - 1; i >= 0; i--)
                {
                    writer << " + (((tid / " << input_strides[i] << ") % " << input_shape[i + 1]
                           << ") * " << 1 + pad_interior[i] << ") * " << output_strides[i];
                }
                writer << ";\n";
                writer << "out[idx] = in[tid];\n";
            }
            writer.block_end();
        }
        writer.block_end();

        compiled_kernel = m_ctx->compiled_kernel_pool->set(hash, writer.get_code());
    }
    std::unique_ptr<gpu::primitive> pad;

    // if the pad value is statically provided, the kernel call signature is different
    if (pad_value == "") // pad value provided at runtime (dynamic)
    {
        pad.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            void* args_list[] = {&inputs[1], &inputs[0], &outputs[0]};
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list,
                                          0)); // arguments
            debug_sync();
        }});
    }
    else // pad value provided at compile time (static)
    {
        pad.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            void* args_list[] = {&inputs[0], &outputs[0]};
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list,
                                          0)); // arguments
            debug_sync();
        }});
    }

    primitive_index = this->m_primitive_emitter->insert(std::move(pad));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_pad_dynamic(const std::array<std::string, 2>& dtypes,
                                                    NVShape input_shape,
                                                    NVShape output_shape,
                                                    NVShape padding_below,
                                                    NVShape padding_interior)
{
    uint32_t rank = static_cast<uint32_t>(input_shape.size());
    std::stringstream kernel_name;
    kernel_name << "pad_dynamic_" << join(dtypes, "_") << rank;

    std::string hash = kernel_name.str() + "pad_i" + join(input_shape, "_") + "pad_o" +
                       join(output_shape) + "_pb" + join(padding_below, "_") + "_pi" +
                       join(padding_interior, "_");
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    NVShape pad_below(input_shape.size(), 0);
    NVShape pad_interior(input_shape.size(), 1);

    int64_t i = padding_below.size() - 1;
    int64_t j = input_shape.size() - 1;
    for (; i >= 0; i--, j--)
    {
        pad_below[j] = padding_below[i];
        pad_interior[j] = padding_interior[i];
    }

    NVShape input_strides = row_major_strides(input_shape);
    NVShape output_strides = row_major_strides(output_shape);

    auto args = m_primitive_emitter->add_kernel_args();
    args.add_placeholder(dtypes[0], "in")
        .add_placeholder(dtypes[1], "out")
        .add("input_strides", input_strides)
        .add("output_strides", output_strides)
        .add("padding_below", pad_below)
        .add("padding_interior", pad_interior)
        .add("n", nthreads);

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_pad_dynamic_op(writer, kernel_name.str(), args, dtypes, rank);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    // create the launch primitive
    std::unique_ptr<gpu::primitive> pad_dynamic(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void** args_list = args.resolve_placeholder(0, &inputs[0])
                                   .resolve_placeholder(1, &outputs[0])
                                   .get_argument_list();

            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list,
                                          0)); // arguments
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pad_dynamic));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}
size_t runtime::gpu::CUDAEmitter::build_reshape(const std::array<std::string, 2>& dtypes,
                                                NVShape input_shape,
                                                NVShape input_order)
{
    auto rank = input_shape.size();
    std::stringstream kernel_name;
    kernel_name << "reshape_" << join(dtypes, "_") << "_r_" << rank;

    std::string hash =
        kernel_name.str() + "_i_" + join(input_shape, "_") + "_o_" + join(input_order, "_");
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_reshape_op(writer, kernel_name.str(), dtypes, rank);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    NVShape input_strides = row_major_strides(input_shape);
    NVShape output_strides(rank);
    NVShape trans_strides(rank);
    int stride = 1;
    for (int64_t i = rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= input_shape[input_order[i]];
    }
    for (int64_t i = 0; i < rank; i++)
    {
        trans_strides[input_order[i]] = output_strides[i];
    }

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t idx_input_strides =
        allocator.reserve_argspace(input_strides.data(), input_strides.size() * sizeof(uint32_t));
    size_t idx_trans_strides =
        allocator.reserve_argspace(trans_strides.data(), trans_strides.size() * sizeof(uint32_t));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        void* param_input_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_input_strides);
        void* param_trans_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_trans_strides);
        std::vector<void*> args_list{
            &inputs[0], &outputs[0], &param_input_strides, &param_trans_strides, &nthreads};

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      aligned_grid_size_x,
                                      1,
                                      1, // grid dim
                                      block_size_x,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_slice(const std::array<std::string, 2>& dtypes,
                                              NVShape input_shape,
                                              NVShape lower_bounds,
                                              NVShape slice_strides,
                                              NVShape output_shape)
{
    std::stringstream kernel_name;
    kernel_name << "slice_" << join(dtypes, "_") << "_r_" << output_shape.size();

    std::string hash = kernel_name.str() + "_i_" + join(input_shape, "_") + "_o_" +
                       join(output_shape, "_") + "_lb_" + join(lower_bounds, "_") + "_ss_" +
                       join(slice_strides, "_");
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_slice_op(writer, kernel_name.str(), dtypes, output_shape.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    NVShape output_strides = row_major_strides(output_shape);
    NVShape input_strides = row_major_strides(input_shape);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t idx_input_strides =
        allocator.reserve_argspace(input_strides.data(), input_strides.size() * sizeof(uint32_t));
    size_t idx_output_strides =
        allocator.reserve_argspace(output_strides.data(), output_strides.size() * sizeof(uint32_t));
    size_t idx_lower_bounds =
        allocator.reserve_argspace(lower_bounds.data(), lower_bounds.size() * sizeof(uint32_t));
    size_t idx_slice_strides =
        allocator.reserve_argspace(slice_strides.data(), slice_strides.size() * sizeof(uint32_t));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        void* param_input_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_input_strides);
        void* param_output_strides =
            runtime::gpu::invoke_memory_primitive(m_ctx, idx_output_strides);
        void* param_lower_bounds = runtime::gpu::invoke_memory_primitive(m_ctx, idx_lower_bounds);
        void* param_slice_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_slice_strides);
        std::vector<void*> args_list{&inputs[0],
                                     &outputs[0],
                                     &param_input_strides,
                                     &param_lower_bounds,
                                     &param_slice_strides,
                                     &param_output_strides,
                                     &nthreads};

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      aligned_grid_size_x,
                                      1,
                                      1, // grid dim
                                      block_size_x,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_reverse_sequence(const std::array<std::string, 3>& dtypes,
                                                         NVShape input_shape0,
                                                         NVShape input_shape1,
                                                         NVShape output_shape,
                                                         size_t batch_axis,
                                                         size_t sequence_axis)
{
    std::stringstream kernel_name;
    kernel_name << "reverse_sequence_" << join(dtypes, "_") << "_bi_" << batch_axis << "_si_"
                << sequence_axis << "_r_" << output_shape.size();

    std::string hash = kernel_name.str() + "_i" + join(input_shape0, "_") + "_i" +
                       join(input_shape1, "_") + "_o" + join(output_shape);
    // For backwards compatability we currently use two unordered maps
    // 1. one looks up the compiled cuda kernel (CudaFunctionPool)
    // 2. the other looks to see if this kernel is already in the primitive list

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_reverse_sequence_op(
            writer, kernel_name.str(), dtypes, batch_axis, sequence_axis, output_shape.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    NVShape output_strides = row_major_strides(output_shape);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t idx_output_shape =
        allocator.reserve_argspace(output_shape.data(), output_shape.size() * sizeof(uint32_t));
    size_t idx_output_strides =
        allocator.reserve_argspace(output_strides.data(), output_strides.size() * sizeof(uint32_t));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        void* param_output_shape = runtime::gpu::invoke_memory_primitive(m_ctx, idx_output_shape);
        void* param_output_strides =
            runtime::gpu::invoke_memory_primitive(m_ctx, idx_output_strides);
        std::vector<void*> args_list{&inputs[0],
                                     &inputs[1],
                                     &outputs[0],
                                     &param_output_shape,
                                     &param_output_strides,
                                     &nthreads};

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      aligned_grid_size_x,
                                      1,
                                      1, // grid dim
                                      block_size_x,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_1d_max_pool(const std::array<std::string, 2>& dtypes,
                                                    NVShape input_shape,
                                                    NVShape output_shape,
                                                    size_t window_width,
                                                    size_t window_stride)
{
    auto input_width = input_shape.back();
    auto output_width = output_shape.back();

    std::string kernel_name = "maxpool_" + join(dtypes, "_") + "_iw" + std::to_string(input_width) +
                              "_ow" + std::to_string(output_width) + "_ww" +
                              std::to_string(window_width) + "_wst" + std::to_string(window_stride);
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    size_t nthreads = shape_size(output_shape);
    std::string hash = kernel_name + "_n" + std::to_string(nthreads);
    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(hash);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::get_max_pool_1d(
            writer, kernel_name, dtypes, input_width, output_width, window_width, window_stride);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    std::unique_ptr<gpu::primitive> pool(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* args_list[] = {&inputs[0], &outputs[0], &nthreads};
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list,
                                          0)); // arguments
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pool));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

pooling_op_shape
    avgpool_shape(NVShape in, NVShape out, NVShape window, NVShape strides, NVShape pad)
{
    pooling_op_shape shape;
    shape.N = in[0];
    shape.C = in[1];
    shape.K = shape.C; // pooling feature maps is
    shape.J = shape.C; // not currently supported
    if (in.size() == 3)
    {
        shape.D = 1;
        shape.H = 1;
        shape.W = in[2];
        shape.M = 1;
        shape.P = 1;
        shape.Q = out[2];
        shape.T = 1;
        shape.R = 1;
        shape.S = window[0];
        shape.STRIDE_D = 0;
        shape.STRIDE_H = 0;
        shape.STRIDE_W = strides[0];
        shape.PAD_D = 0;
        shape.PAD_H = 0;
        shape.PAD_W = pad[0];
    }
    else if (in.size() == 4)
    {
        shape.D = 1;
        shape.H = in[2];
        shape.W = in[3];
        shape.M = 1;
        shape.P = out[2];
        shape.Q = out[3];
        shape.T = 1;
        shape.R = window[0];
        shape.S = window[1];
        shape.STRIDE_D = 0;
        shape.STRIDE_H = strides[0];
        shape.STRIDE_W = strides[1];
        shape.PAD_D = 0;
        shape.PAD_H = pad[0];
        shape.PAD_W = pad[1];
    }
    else if (in.size() == 5)
    {
        shape.D = in[2];
        shape.H = in[3];
        shape.W = in[4];
        shape.M = out[2];
        shape.P = out[3];
        shape.Q = out[4];
        shape.T = window[0];
        shape.R = window[1];
        shape.S = window[2];
        shape.STRIDE_D = strides[0];
        shape.STRIDE_H = strides[1];
        shape.STRIDE_W = strides[2];
        shape.PAD_D = pad[0];
        shape.PAD_H = pad[1];
        shape.PAD_W = pad[2];
    }
    else
    {
        throw std::runtime_error("AvgPool currently supports up to 3 spatial dimensions.");
    }
    return shape;
}

size_t runtime::gpu::CUDAEmitter::build_avg_pool(const std::array<std::string, 2>& dtypes,
                                                 NVShape input_shape,
                                                 NVShape output_shape,
                                                 NVShape window_shape,
                                                 NVShape window_stride,
                                                 NVShape padding_below,
                                                 bool include_pad)
{
    // assumes NCDHW format
    pooling_op_shape shape =
        avgpool_shape(input_shape, output_shape, window_shape, window_stride, padding_below);

    std::string kernel_name = "avgpool";
    std::stringstream ss;
    ss << kernel_name << "_s" << shape << "_st" << join(window_stride, "_") << "_ip"
       << int(include_pad);
    auto hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // if the kernel has not been compiled, build it
    kernel_name += "_ip" + std::to_string(int(include_pad));
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        writer << include_helpers();
        CudaKernelBuilder::get_avg_pool(writer, kernel_name, dtypes, include_pad);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    // precompute for fast constant memory access
    int HW = shape.H * shape.W;
    int DHW = shape.D * HW;
    int CDHW = shape.C * DHW;
    int PQ = shape.P * shape.Q;
    int MPQ = shape.M * PQ;
    int KMPQ = shape.K * MPQ;
    int RS = shape.R * shape.S;
    int TRS = shape.T * RS;

    // precompute magic numbers and shifts for fast integer division
    int magic_N;
    int shift_N;
    std::tie(magic_N, shift_N) = idiv_magic_u64(shape.N);
    int magic_P;
    int shift_P;
    std::tie(magic_P, shift_P) = idiv_magic_u64(shape.P);
    int magic_S;
    int shift_S;
    std::tie(magic_S, shift_S) = idiv_magic_u64(shape.S);
    int magic_RS;
    int shift_RS;
    std::tie(magic_RS, shift_RS) = idiv_magic_u64(RS);

    // TODO: blending factors are not currently implemented
    float alpha = 1.0f;
    float beta = 0.0f;

    std::unique_ptr<gpu::primitive> pool(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* args_list[] = {&inputs[0],
                                 &outputs[0],
                                 &alpha,
                                 &beta,
                                 &shape.N,
                                 &shape.C,
                                 &shape.D,
                                 &shape.H,
                                 &shape.W,
                                 &HW,
                                 &DHW,
                                 &CDHW,
                                 &magic_N,
                                 &shift_N,
                                 &shape.P,
                                 &shape.Q,
                                 &magic_P,
                                 &shift_P,
                                 &PQ,
                                 &MPQ,
                                 &KMPQ,
                                 &shape.S,
                                 &RS,
                                 &TRS,
                                 &magic_S,
                                 &shift_S,
                                 &magic_RS,
                                 &shift_RS,
                                 &shape.STRIDE_D,
                                 &shape.STRIDE_H,
                                 &shape.STRIDE_W,
                                 &shape.PAD_D,
                                 &shape.PAD_H,
                                 &shape.PAD_W};
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          shape.Q,
                                          shape.M * shape.P,
                                          shape.N * shape.K,
                                          32,
                                          1,
                                          1,
                                          0,
                                          NULL,
                                          args_list,
                                          0));
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pool));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_elementwise_n_to_1(const std::vector<std::string>& dtypes,
                                                           NVShape tensor_shape,
                                                           const char* op,
                                                           const char* kernel)
{
    // kernel_name is used to check if the cuda kernel has been previously compiled
    std::stringstream kernel_name;
    kernel_name << "ew"
                << "_" << op << "_" << join(dtypes, "_");

    // hash is used to check if the emitted primitive already exists
    std::stringstream ss;
    ss << kernel_name.str() << "_s" << join(tensor_shape, "_");
    auto hash = ss.str();

    // if the primitive exists, we are done
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        if (kernel)
        {
            CudaKernelBuilder::get_device_helper(writer, op, kernel, dtypes);
        }

        CudaKernelBuilder::get_elementwise_op(writer, kernel_name.str(), op, dtypes);

        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }
    uint32_t nthreads = static_cast<uint32_t>(shape_size(tensor_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 512;
    int num_SMs;
    CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
    uint32_t aligned_grid_size_x = fmin(num_SMs * 32, align_to_block_size(nthreads, block_size_x));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> ew(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            std::vector<void*> args_list;
            for (auto i = 0u; i < dtypes.size() - 1; i++)
            {
                args_list.push_back(&inputs[i]);
            }
            args_list.push_back(&outputs[0]);
            args_list.push_back(&nthreads);
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          args_list.data(),
                                          0)); // arguments
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(ew));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_primitive(const op::MaxPool* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto& input_shape = args[0].get_shape();
    auto& result_shape = out[0].get_shape();
    auto padding_below = node->get_padding_below();
    auto padding_above = node->get_padding_above();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "max_pool_" << runtime::gpu::kernel::emit_type_string(node) << "_i"
       << join(input_shape, "_") << "_o" << join(result_shape, "_") << "_ws"
       << join(node->get_window_shape(), "_") << "_wst"
       << join(node->get_window_movement_strides(), "_") << "_pb" << join(padding_below, "_")
       << "_pb" << join(padding_above, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    /// assymetric padding detection
    bool pad_required = false;
    auto shape_to_pool =
        runtime::gpu::get_padded_shape(input_shape, padding_below, padding_above, {});
    if (shape_to_pool != input_shape)
    {
        pad_required = true;
    }

    pad_required = pad_required && (padding_below != padding_above);
    // asymetric padding
    size_t idx_workspace = std::numeric_limits<size_t>::max();
    size_t pad_index = std::numeric_limits<size_t>::max();
    if (pad_required)
    {
        auto temp_size = shape_size(shape_to_pool) * args[0].get_element_type().size();
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();
        idx_workspace = allocator.reserve_workspace(temp_size);

        auto pad_value = TypeInfo::Get(args[0].get_element_type())->lowest();

        pad_index = build_pad({{input_type, output_type}},
                              input_shape,
                              shape_to_pool,
                              padding_below,
                              padding_above,
                              Shape{},
                              pad_value);
    }
    /// end asymmetric padding detection

    size_t max_pool_index = build_1d_max_pool({{input_type, output_type}},
                                              input_shape,
                                              result_shape,
                                              node->get_window_shape().back(),
                                              node->get_window_movement_strides().back());

    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            if (idx_workspace != std::numeric_limits<size_t>::max() &&
                pad_index != std::numeric_limits<size_t>::max())
            {
                // void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                // gpu::invoke_primitive(m_ctx,
                //                       pad_dynamic_index,
                //                       std::vector<void*>{inputs[0]}.data(),
                //                       std::vector<void*>{pad_buffer}.data());

                // gpu::invoke_primitive(
                //     m_ctx, conv_index, std::vector<void*>{pad_buffer, inputs[1]}.data(), outputs);

                void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                gpu::invoke_primitive(m_ctx,
                                      pad_index,
                                      std::vector<void*>{inputs[0]}.data(),
                                      std::vector<void*>{pad_buffer}.data());
                gpu::invoke_primitive(
                    m_ctx, max_pool_index, std::vector<void*>{pad_buffer}.data(), outputs);
            }
            else
            {
                gpu::invoke_primitive(m_ctx, max_pool_index, inputs, outputs);
            }
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_softmax_divide(const std::vector<std::string>& dtypes,
                                                       NVShape input_shape,
                                                       NVShape reduce_shape,
                                                       std::vector<size_t> axes_flag)
{
    std::string kernel_name =
        "softmax_divide_" + join(dtypes, "_") + "_axes_" + join(axes_flag, "_");
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    size_t nthreads = shape_size(input_shape);

    std::string hash = kernel_name + "_n" + join(input_shape, "_") + join(reduce_shape, "_") +
                       std::to_string(nthreads);
    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(hash);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        writer << include_helpers();
        CudaKernelBuilder::get_softmax_divide_op(
            writer, kernel_name, dtypes, axes_flag, input_shape.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    NVShape input_strides = row_major_strides(input_shape);
    NVShape reduce_strides = row_major_strides(reduce_shape);

    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();

    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    std::unique_ptr<gpu::primitive> pool(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            std::vector<void*> arg_list;
            arg_list.push_back(&inputs[0]);
            arg_list.push_back(&inputs[1]);
            arg_list.push_back(&outputs[0]);
            for (size_t i = 0; i < input_strides.size(); i++)
            {
                arg_list.push_back(&input_strides[i]);
            }
            for (size_t i = 0; i < reduce_strides.size(); i++)
            {
                arg_list.push_back(&reduce_strides[i]);
            }
            arg_list.push_back(&nthreads);
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1, // grid dim
                                          block_size_x,
                                          1,
                                          1, // block dim
                                          0,
                                          NULL, // shared mem and stream
                                          arg_list.data(),
                                          0)); // arguments
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pool));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_reduce(const std::vector<std::string>& dtypes,
                                               NVShape input_shape,
                                               NVShape reduce_axis,
                                               const char* op,
                                               const char* kernel)
{
    // assumes NC{d1,...,dn} format
    std::string kernel_name = "reduce_" + join(dtypes, "_") + "_ri_" +
                              std::to_string(input_shape.size()) + "_rr_" +
                              std::to_string(reduce_axis.size());
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    std::stringstream ss;
    ss << kernel_name << "_s_" << join(input_shape, "_") << "_axis_" << join(reduce_axis, "_");
    auto hash = ss.str();
    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    size_t rank = input_shape.size();
    size_t reduce_rank = reduce_axis.size();
    size_t out_rank = rank - reduce_rank;
    NVShape reduce_flag(rank, 0);
    for (auto a : reduce_axis)
    {
        reduce_flag[a] = 1;
    }
    NVShape output_shape;
    NVShape non_reduce_strides;
    NVShape reduce_shape;
    NVShape reduce_strides;
    NVShape input_strides = row_major_strides(input_shape);
    for (int i = 0; i < rank; i++)
    {
        if (reduce_flag[i] != 0)
        {
            reduce_shape.push_back(input_shape[i]);
            reduce_strides.push_back(input_strides[i]);
        }
        else
        {
            non_reduce_strides.push_back(input_strides[i]);
            output_shape.push_back(input_shape[i]);
        }
    }
    NVShape output_strides = row_major_strides(output_shape);

    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    auto args = m_primitive_emitter->add_kernel_args();
    args.add_placeholder(dtypes[0], "in")
        .add_placeholder(dtypes[1], "out")
        .add("out_strides", output_strides)
        .add("non_reduce_strides", non_reduce_strides)
        .add("reduce_shape", reduce_shape)
        .add("reduce_strides", reduce_strides)
        .add("nthreads", nthreads);

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        writer << include_helpers();
        if (kernel)
        {
            CudaKernelBuilder::get_device_helper(
                writer, op, kernel, {{dtypes[0], dtypes[0], dtypes[1]}});
        }
        runtime::gpu::CudaKernelBuilder::get_reduce_op(
            writer, kernel_name, args, dtypes, op, out_rank, reduce_rank);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    std::unique_ptr<gpu::primitive> reduce(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void** args_list = args.resolve_placeholder(0, &inputs[0])
                                   .resolve_placeholder(1, &outputs[0])
                                   .get_argument_list();

            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1,
                                          block_size_x,
                                          1,
                                          1,
                                          0,
                                          NULL,
                                          args_list,
                                          0));
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(reduce));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_primitive(const op::Softmax* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape = args[0].get_shape();
    auto axes = node->get_axes();

    std::stringstream ss;
    ss << "softmax_" << runtime::gpu::kernel::emit_type_string(node) << "_s"
       << join(input_shape, "_") << "_ra" << join(axes, "_");
    auto hash = ss.str();

    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // build composite primitive
    auto& cudnn_emitter = m_primitive_emitter->get_cudnn_emitter();

    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    auto reduced_shape = input_shape;
    std::vector<size_t> axes_flag(input_shape.size(), 0);
    for (auto const& axis : axes)
    {
        reduced_shape[axis] = 1;
        axes_flag[axis] = 1;
    }
    size_t reduced_size = shape_size(reduced_shape);
    size_t tensor_size = shape_size(input_shape);
    size_t type_size = out[0].get_element_type().size();

    size_t reduce_buffer_idx = allocator.reserve_workspace(reduced_size * type_size);

    // exponentiate with fused sum reduction to calculate softmax denominator
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    auto exp_index = build_elementwise<ngraph::op::Exp>({{input_type, output_type}}, input_shape);
    auto reduce_index = cudnn_emitter->build_reduce_forward(
        CUDNN_REDUCE_TENSOR_ADD, output_type, input_shape, axes);
    size_t divide_index = build_softmax_divide(
        std::vector<std::string>(3, output_type), input_shape, reduced_shape, axes_flag);

    if (reduced_size == tensor_size)
    {
        // the result should be all set to 1.
        // TODO: add memset
        std::unique_ptr<gpu::primitive> kernel_launch(
            new gpu::primitive{[=](void** inputs, void** outputs) mutable {
                runtime::gpu::invoke_primitive(
                    m_ctx, divide_index, std::vector<void*>{inputs[0], inputs[0]}.data(), outputs);
            }});

        primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    }
    else
    {
        std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](
            void** inputs, void** outputs) mutable {
            void* reduce_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, reduce_buffer_idx);
            runtime::gpu::invoke_primitive(m_ctx, exp_index, inputs, outputs);
            runtime::gpu::invoke_primitive(
                m_ctx, reduce_index, outputs, std::vector<void*>{reduce_buffer}.data());
            runtime::gpu::invoke_primitive(
                m_ctx, divide_index, std::vector<void*>{outputs[0], reduce_buffer}.data(), outputs);
        }});

        primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    }
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t
    runtime::gpu::CUDAEmitter::build_fused_ew_to_collective(const std::vector<std::string>& dtypes,
                                                            NVShape tensor_shape,
                                                            const std::set<size_t>& reduced_tensors,
                                                            const std::set<size_t>& axes,
                                                            const char* op,
                                                            const char* kernel,
                                                            const char* reduce_op,
                                                            bool save_elementwise)
{
    // kernel_name is used to check if the cuda kernel has been previously compiled
    std::stringstream kernel_name;
    kernel_name << "ew_collective"
                << "_" << op << "_" << join(dtypes, "_") << "_" << reduce_op << "_r"
                << tensor_shape.size() << "_rt" << join(reduced_tensors, "_")
                // multi-output op
                << "_mo" << int(save_elementwise);

    // hash is used to check if the emitted primitive already exists
    std::stringstream ss;
    ss << kernel_name.str() << "_s" << join(tensor_shape, "_") << "_ra" << join(axes, "_");
    auto hash = ss.str();

    // if the primitive exists, we are done
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // calculate strides
    NVShape strides = row_major_strides(tensor_shape);
    // precacluate invariants for integer division via multiplication
    std::vector<int> stride_magic;
    std::vector<int> stride_shift;
    for (int i = 0; i < strides.size(); i++)
    {
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(strides[i]);
        stride_magic.push_back(magic);
        stride_shift.push_back(shift);
    }
    // calculate reduced tensor strides with 0s inserted for reduced axes
    NVShape reduced_shape = tensor_shape;
    for (auto const& axis : axes)
    {
        reduced_shape[axis] = 1;
    }
    NVShape reduced_strides = row_major_strides(reduced_shape);
    for (auto const& axis : axes)
    {
        reduced_strides[axis] = 0;
    }

    size_t nthreads = shape_size(tensor_shape);
    constexpr const int nthreads_per_block = 32;
    int nblocks = 1 + ((static_cast<int>(nthreads) - 1) / nthreads_per_block);

    auto args = this->m_primitive_emitter->add_kernel_args();
    for (auto i = 0u; i < dtypes.size() - 1; i++)
    {
        args.add_placeholder(dtypes[i], "in" + std::to_string(i));
    }
    args.add_placeholder(dtypes.back(), "out0");
    if (save_elementwise)
    {
        args.add_placeholder(dtypes.back(), "out1");
    }

    args.add("strides", strides)
        .add("stride_magic", stride_magic)
        .add("stride_shift", stride_shift)
        .add("reduced_strides", reduced_strides)
        .add("nthreads", nthreads);

    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        writer << include_helpers();
        if (kernel)
        {
            CudaKernelBuilder::get_device_helper(writer, op, kernel, dtypes);
        }
        CudaKernelBuilder::get_ew_collective_op(writer,
                                                kernel_name.str(),
                                                args,
                                                op,
                                                reduce_op,
                                                dtypes,
                                                reduced_tensors,
                                                save_elementwise,
                                                tensor_shape.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    // TODO: check if mutable is necessary
    std::unique_ptr<gpu::primitive> ew_collective(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            for (auto i = 0u; i < dtypes.size() - 1; i++)
            {
                args.resolve_placeholder(i, &inputs[i]);
            }
            args.resolve_placeholder(dtypes.size() - 1, &outputs[0]);
            if (save_elementwise)
            {
                args.resolve_placeholder(dtypes.size(), &outputs[1]);
            }
            void** args_list = args.get_argument_list();

            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          nblocks,
                                          1,
                                          1,
                                          nthreads_per_block,
                                          1,
                                          1,
                                          0,
                                          NULL,
                                          args_list,
                                          0));
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(ew_collective));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_reduce_window(const OpName op_name,
                                                      const std::vector<std::string>& dtypes,
                                                      NVShape input_shape,
                                                      NVShape output_shape,
                                                      NVShape reduce_window_shape,
                                                      NVShape reduce_window_strides)
{
    const char* op = NULL;
    const char* kernel = NULL;
    switch (op_name)
    {
    case OpName::add:
        op = CudaOpMap<ngraph::op::Add>::op;
        kernel = CudaOpMap<ngraph::op::Add>::math_kernel;
        break;
    case OpName::multiply:
        op = CudaOpMap<ngraph::op::Multiply>::op;
        kernel = CudaOpMap<ngraph::op::Multiply>::math_kernel;
        break;
    case OpName::minimum:
        op = CudaOpMap<ngraph::op::Minimum>::op;
        kernel = CudaOpMap<ngraph::op::Minimum>::math_kernel;
        break;
    case OpName::maximum:
        op = CudaOpMap<ngraph::op::Maximum>::op;
        kernel = CudaOpMap<ngraph::op::Maximum>::math_kernel;
    }
    // kernel_name is used to check if the cuda kernel has been previously compiled
    size_t rank = input_shape.size();
    std::stringstream kernel_name;
    kernel_name << "reduce_window"
                << "_" << op << "_" << join(dtypes, "_") << rank;

    // hash is used to check if the emitted primitive already exists
    std::stringstream ss;
    ss << kernel_name.str() << "_s" << join(output_shape, "_");
    auto hash = ss.str();

    // if the primitive exists, we are done
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // check if the kernel has already been compiled. if so, create
    // a launch primitive for it based on the input tensor shape
    // but do not recompile the kernel. otherwise, do it all:
    // recompile the kernel and then create the primitive
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name.str());
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        if (kernel)
        {
            CudaKernelBuilder::get_device_helper(writer, op, kernel, dtypes);
        }
        CudaKernelBuilder::get_reduce_window_op(writer, kernel_name.str(), op, dtypes, rank);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name.str(), writer.get_code());
    }

    size_t nthreads = shape_size(output_shape);
    NVShape input_strides = row_major_strides(input_shape);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();

    // (lazy) allocation for kernel arguments
    size_t idx_input_strides = allocator.reserve_argspace(input_strides.data(), rank * sizeof(int));
    size_t idx_output_shape = allocator.reserve_argspace(output_shape.data(), rank * sizeof(int));
    size_t idx_reduce_window_shape =
        allocator.reserve_argspace(reduce_window_shape.data(), rank * sizeof(int));
    size_t idx_reduce_window_strides =
        allocator.reserve_argspace(reduce_window_strides.data(), rank * sizeof(int));

    // create the launch primitive
    std::unique_ptr<gpu::primitive> f(new gpu::primitive{[=](void** inputs,
                                                             void** outputs) mutable {
        void* param_input_strides = runtime::gpu::invoke_memory_primitive(m_ctx, idx_input_strides);
        void* param_output_shape = runtime::gpu::invoke_memory_primitive(m_ctx, idx_output_shape);
        void* param_reduce_window_shape =
            runtime::gpu::invoke_memory_primitive(m_ctx, idx_reduce_window_shape);
        void* param_reduce_window_strides =
            runtime::gpu::invoke_memory_primitive(m_ctx, idx_reduce_window_strides);

        std::vector<void*> args_list(7, NULL);
        args_list[0] = &inputs[0];
        args_list[1] = &outputs[0];
        args_list[2] = &param_input_strides;
        args_list[3] = &param_output_shape;
        args_list[4] = &param_reduce_window_shape;
        args_list[5] = &param_reduce_window_strides;
        args_list[6] = &nthreads;

        CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                      static_cast<uint32_t>(nthreads),
                                      1,
                                      1, // grid dim
                                      1,
                                      1,
                                      1, // block dim
                                      0,
                                      NULL, // shared mem and stream
                                      args_list.data(),
                                      0)); // arguments
        debug_sync();

    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(f));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_broadcast(const std::array<std::string, 2>& dtypes,
                                                  NVShape result_shape,
                                                  const std::set<size_t>& reduce_axes)
{
    // assumes NC{d1,...,dn} format
    std::string kernel_name =
        "broadcast_" + join(dtypes, "_") + "_r" + std::to_string(result_shape.size());
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    std::stringstream ss;
    ss << kernel_name << "_s" << join(result_shape, "_") << "_rs" << join(reduce_axes, "_");
    auto hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // calculate strides
    NVShape strides = row_major_strides(result_shape);
    // precacluate invariants for integer division via multiplication
    std::vector<int> stride_magic;
    std::vector<int> stride_shift;
    for (int i = 0; i < strides.size(); i++)
    {
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(strides[i]);
        stride_magic.push_back(magic);
        stride_shift.push_back(shift);
    }
    // calculate reduced tensor strides with 0s inserted for reduced axes
    NVShape reduced_shape = result_shape;
    for (auto const& axis : reduce_axes)
    {
        reduced_shape[axis] = 1;
    }
    NVShape reduced_strides = row_major_strides(reduced_shape);
    for (auto const& axis : reduce_axes)
    {
        reduced_strides[axis] = 0;
    }

    // TODO: blending factors are not currently implemented
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t nthreads = shape_size(result_shape);
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    auto args = this->m_primitive_emitter->add_kernel_args();
    args.add_placeholder(dtypes[0], "in")
        .add_placeholder(dtypes[1], "out")
        .add("strides", strides)
        .add("stride_magic", stride_magic)
        .add("stride_shift", stride_shift)
        .add("reduced_strides", reduced_strides)
        .add("alpha", alpha)
        .add("beta", beta)
        .add("nthreads", nthreads);

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        writer << include_helpers();
        runtime::gpu::CudaKernelBuilder::get_broadcast_op(
            writer, kernel_name, args, result_shape.size());
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    std::unique_ptr<gpu::primitive> broadcast(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void** args_list = args.resolve_placeholder(0, &inputs[0])
                                   .resolve_placeholder(1, &outputs[0])
                                   .get_argument_list();

            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          aligned_grid_size_x,
                                          1,
                                          1,
                                          block_size_x,
                                          1,
                                          1,
                                          0,
                                          NULL,
                                          args_list,
                                          0));
            debug_sync();
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(broadcast));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_primitive(const op::Convolution* node)
{
    std::stringstream ss;
    ss << "convolution_fprop_" << runtime::gpu::kernel::emit_type_string(node);

    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape = args[0].get_shape();
    auto filter_shape = args[1].get_shape();
    auto output_shape = out[0].get_shape();
    auto tensor_size = input_shape.size();

    // primitive cache parameters
    ss << "_s" << join(input_shape, "_") << "_pb" << join(node->get_padding_below(), "_") << "_pi"
       << join(node->get_data_dilation_strides(), "_") << "_fs" << join(filter_shape, "_") << "_fst"
       << join(node->get_window_movement_strides(), "_") << "_fdi"
       << join(node->get_window_dilation_strides(), "_");

    auto hash = ss.str();

    // check if the requested primtive is already built
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // Reshape from NC{d1,..,dn} -> C{d1,...,dn}N
    // and from KC{df1,...,dfn} -> C{df1,...,dfn}N.

    // TODO: This should be done via a pass similar to
    // what is done for convolution in the IA transformer
    // c.f runtime/cpu/pass/cpu_layout.cpp

    GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();
    size_t transposed_data_idx =
        allocator.reserve_workspace(shape_size(input_shape) * args[0].get_element_type().size());
    size_t transposed_filter_idx =
        allocator.reserve_workspace(shape_size(filter_shape) * args[1].get_element_type().size());
    size_t transposed_output_idx =
        allocator.reserve_workspace(shape_size(output_shape) * out[0].get_element_type().size());

    NVShape input_order;
    for (int i = 1; i <= tensor_size; i++)
    {
        input_order.push_back(i % tensor_size);
    }

    size_t reshape_data_index = build_reshape(
        {{args[0].get_element_type().c_type_string(), args[0].get_element_type().c_type_string()}},
        input_shape,
        input_order);

    size_t reshape_filter_index = build_reshape(
        {{args[1].get_element_type().c_type_string(), args[1].get_element_type().c_type_string()}},
        filter_shape,
        input_order);

    // local helper to reshape tensor shape objects
    auto reshape = [](const Shape& shape, const NVShape& order) {
        Shape output(shape.size(), 0);
        for (size_t i = 0; i < shape.size(); i++)
        {
            output[i] = shape[order[i]];
        }
        return output;
    };

    // reorder axes of the input shape (NC{d_1,...,d_n} -> C{d_1,...,d_n}N)
    input_shape = reshape(input_shape, input_order);
    // reorder axes of the filter shape (KC{df_1,...,df_n} -> C{df_1,...,df_n}K)
    filter_shape = reshape(filter_shape, input_order);
    // reorder axes of the output shape (NK{do_1,...,do_n} -> K{do_1,...,do_n}N)
    output_shape = reshape(output_shape, input_order);

    size_t conv_index = build_convolution({{args[0].get_element_type().c_type_string(),
                                            args[1].get_element_type().c_type_string(),
                                            out[0].get_element_type().c_type_string()}},
                                          input_shape,
                                          filter_shape,
                                          output_shape,
                                          node->get_window_movement_strides(),
                                          node->get_window_dilation_strides(),
                                          node->get_data_dilation_strides(),
                                          node->get_padding_below());

    // reshape output tensor (K{do_1,...,do_n}N -> NK{do_1,...,do_n})
    input_order.clear();
    input_order.push_back(static_cast<int>(tensor_size - 1));
    for (int i = 0; i < tensor_size - 1; i++)
    {
        input_order.push_back(i);
    }

    size_t reshape_output_index = build_reshape(
        {{args[1].get_element_type().c_type_string(), args[1].get_element_type().c_type_string()}},
        output_shape,
        input_order);

    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* data = gpu::invoke_memory_primitive(m_ctx, transposed_data_idx);
            void* filter = gpu::invoke_memory_primitive(m_ctx, transposed_filter_idx);
            void* output = gpu::invoke_memory_primitive(m_ctx, transposed_output_idx);
            gpu::invoke_primitive(m_ctx,
                                  reshape_data_index,
                                  std::vector<void*>{inputs[0]}.data(),
                                  std::vector<void*>{data}.data());
            gpu::invoke_primitive(m_ctx,
                                  reshape_filter_index,
                                  std::vector<void*>{inputs[1]}.data(),
                                  std::vector<void*>{filter}.data());
            gpu::invoke_primitive(m_ctx,
                                  conv_index,
                                  std::vector<void*>{data, filter}.data(),
                                  std::vector<void*>{output}.data());
            gpu::invoke_primitive(m_ctx,
                                  reshape_output_index,
                                  std::vector<void*>{output}.data(),
                                  std::vector<void*>{outputs[0]}.data());
        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_primitive(const op::ReplaceSlice* node, bool in_place_op)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto& input_shape = args[0].get_shape();
    auto& replace_shape = args[1].get_shape();
    auto& lower_bounds = node->get_lower_bounds();
    auto& upper_bounds = node->get_upper_bounds();
    auto& slice_strides = node->get_strides();
    Shape slice_shape(upper_bounds.size(), 0);
    std::transform(upper_bounds.begin(),
                   upper_bounds.end(),
                   lower_bounds.begin(),
                   slice_shape.begin(),
                   std::minus<size_t>());
    std::transform(slice_shape.begin(),
                   slice_shape.end(),
                   slice_strides.begin(),
                   slice_shape.begin(),
                   std::divides<size_t>());

    auto input_type = args[0].get_element_type().c_type_string();
    auto replace_type = args[1].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // assumes NC{d1,...,dn} format
    std::string type_str = input_type + "_" + replace_type + "_" + output_type;
    std::replace(type_str.begin(), type_str.end(), ' ', '_');

    std::stringstream ss;
    ss << "rep_slices_" << type_str << "_s" << join(input_shape, "_") << "_ssrc"
       << join(replace_shape, "_") << "_sll" << join(lower_bounds, "_") << "_slu"
       << join(upper_bounds, "_") << "_slst" << join(slice_strides, "_") << in_place_op;
    auto hash = ss.str();

    // check if the requested primtive is already built
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    // calculate strides
    Shape input_strides = row_major_strides(input_shape);
    Shape replace_strides = row_major_strides(replace_shape);

    size_t pad_index = build_pad_dynamic(
        {{input_type, output_type}}, replace_shape, input_shape, lower_bounds, slice_strides);

    if (in_place_op)
    {
        std::unique_ptr<gpu::primitive> kernel_launch(
            new gpu::primitive{[=](void** inputs, void** outputs) mutable {
                runtime::gpu::invoke_primitive(
                    m_ctx, pad_index, std::vector<void*>{inputs[1]}.data(), outputs);
            }});
        primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    }
    else
    {
        size_t nthreads = shape_size(input_shape);
        size_t size = nthreads * args[1].get_element_type().size();
        std::unique_ptr<gpu::primitive> kernel_launch(
            new gpu::primitive{[=](void** inputs, void** outputs) mutable {
                runtime::gpu::cuda_memcpyDtD(outputs[0], inputs[0], size);
                runtime::gpu::invoke_primitive(
                    m_ctx, pad_index, std::vector<void*>{inputs[1]}.data(), outputs);
            }});
        primitive_index = this->m_primitive_emitter->insert(std::move(kernel_launch));
    }
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_convolution(const std::array<std::string, 3>& dtypes,
                                                    NVShape input_shape,
                                                    NVShape filter_shape,
                                                    NVShape output_shape,
                                                    NVShape filter_stride,
                                                    NVShape filter_dilation,
                                                    NVShape input_dilation,
                                                    NVDiff input_pad_below)
{
    // convolution is performed on tensors in the following format
    // input_shape:  C{di_1,...,du_n}N
    // filter_shape: C{df_1,...,df_n}K
    // output_shape: K{do_1,...,do_n}N

    // The basic strategy performed by this kernel is to convert Nd convolution
    // into a single 2D GEMM that can be block multiplied via a hierarchical strategy.
    // The spatial dimensions are squashed into a single column axis and the
    // batch number N and filter number K are the rows of A and B in the 2D GEMM
    // A * B = C, respectively. By keeping N and K in contiguous memory space,
    // coalescing and vectorization is maintained regardless of coordinate access
    // (e.g. data and filter dilation).

    // prerequisits for kernel cacheing and building
    int N = input_shape.back();
    int K = filter_shape.back();
    int filter_size = 1;
    int rank = 0;
    for (int i = 1; i < filter_shape.size() - 1; i++)
    { // skip first and last (non-spatial) dimensions
        filter_size *= filter_shape[i];
        rank++;
    }

    // tiling options are determined by
    // batch size (N) and number of filters (K)
    int reg_tile_size = 1;
    int sm_tile_size = 8;
    // if N is a multiple of 32 use register tiling
    if (N % (sm_tile_size * 4) == 0)
    {
        reg_tile_size = 4;
    }

    // TODO: as each cuda_emitter has a regular structure
    // it would be beneficial to factor these into classes
    // with seperate methods for compiling the kernel, building
    // the primitive, and transfering arguments to device memory

    int C = input_shape.front();
    int input_channel_size = 1;
    int filter_channel_size = 1;
    int output_filter_size = 1;
    for (int i = 1; i < input_shape.size(); i++)
    {
        input_channel_size *= input_shape[i];
        filter_channel_size *= filter_shape[i];
        output_filter_size *= output_shape[i];
    }
    // vector accesses of width `reg_tile_size` are
    // used reducting the effective tensor array size
    input_channel_size /= reg_tile_size;
    filter_channel_size /= reg_tile_size;
    output_filter_size /= reg_tile_size;

    // arguments derived from output tensor
    int output_pixels = 1;
    int output_pixels_magic;
    int output_pixels_shift;
    std::vector<int> output_dim_strides(rank, 0);
    std::vector<int> output_str_magic(rank, 0);
    std::vector<int> output_str_shift(rank, 0);
    for (int64_t i = output_shape.size() - 2; i > 0; i--)
    {
        output_dim_strides[i - 1] = output_pixels;
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(output_pixels);
        output_str_magic[i - 1] = magic;
        output_str_shift[i - 1] = shift;
        output_pixels *= output_shape[i];
    }
    std::tie(output_pixels_magic, output_pixels_shift) = idiv_magic_u64(output_pixels);

    // arguments derived from filter tensor
    int filter_sz = 1;
    std::vector<int> filter_dim_strides(rank, 0);
    std::vector<int> filter_str_magic(rank, 0);
    std::vector<int> filter_str_shift(rank, 0);
    for (int64_t i = filter_shape.size() - 2; i > 0; i--)
    {
        filter_dim_strides[i - 1] = filter_sz;
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(filter_sz);
        filter_str_magic[i - 1] = magic;
        filter_str_shift[i - 1] = shift;
        filter_sz *= filter_shape[i];
    }

    // remaining kernel arguments
    std::vector<int> data_dilation_magic(input_dilation.size(), 0);
    std::vector<int> data_dilation_shift(input_dilation.size(), 0);
    for (int i = 0; i < input_dilation.size(); i++)
    {
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(input_dilation[i]);
        data_dilation_magic[i] = magic;
        data_dilation_shift[i] = shift;
    }
    NVShape input_shape_str = row_major_strides(input_shape);
    float alpha = 1.0f;
    float beta = 0.0f;

    auto args = m_primitive_emitter->add_kernel_args();
    args.add_placeholder(dtypes[0], "in")
        .add_placeholder(dtypes[1], "filter")
        .add_placeholder(dtypes[2], "out")
        .add("alpha", alpha)
        .add("beta", beta)
        .add("N", N)
        .add("C", C)
        .add("K", K)
        .add("input_channel_size", input_channel_size)
        .add("filter_channel_size", filter_channel_size)
        .add("output_filter_size", output_filter_size)
        .add("output_pixels", output_pixels)
        .add("output_pixels_magic", output_pixels_magic)
        .add("output_pixels_shift", output_pixels_shift)
        .add("pad", input_pad_below)
        .add("data_dilation", input_dilation)
        .add("data_dilation_magic", data_dilation_magic)
        .add("data_dilation_shift", data_dilation_shift)
        .add("filter_strides", filter_stride)
        .add("filter_dilation", filter_dilation)
        .add("in_shape", input_shape)
        .add("in_shape_str", input_shape_str)
        .add("out_dim_str", output_dim_strides)
        .add("out_str_magic", output_str_magic)
        .add("out_str_shift", output_str_shift)
        .add("filter_dim_str", filter_dim_strides)
        .add("filter_str_magic", filter_str_magic)
        .add("filter_str_shift", filter_str_shift);

    std::string kernel_name = "convolution_fprop_c_nd_n" + join(dtypes, "_") + "_n" +
                              std::to_string(N) + "_k" + std::to_string(K) + "_fsz" +
                              std::to_string(filter_size) + "_r" + std::to_string(rank);
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    // if the kernel has not been compiled, build it
    auto compiled_kernel = m_ctx->compiled_kernel_pool->get(kernel_name);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        writer << include_helpers();
        runtime::gpu::CudaKernelBuilder::get_convolution_forward(writer,
                                                                 kernel_name,
                                                                 dtypes,
                                                                 args,
                                                                 N,
                                                                 K,
                                                                 rank,
                                                                 filter_size,
                                                                 sm_tile_size,
                                                                 reg_tile_size);
        compiled_kernel = m_ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
    }

    // launch arguments:
    // each output pixel is its own block. if the batch size is greater than reg_tile_size * sm_tile_size, a single
    // output pixel is spread over multiple blocks along the batch axis so that memory coordination is not required
    // each block consists of 2 warps in an 8 x 8 array used for accessing the SM block of the GEMM

    // do_i = output pixel coordinates
    // grid = (do_1*do_2*...*do_N*ceil_div(N, REG_TILE_SIZE*SM_TILE_SIZE), ceil_div(K, REG_TILE_SIZE*SM_TILE_SIZE), 1)
    // block = (8, 8, 1)
    dim3 blocks(output_pixels * idiv_ceil(N, reg_tile_size * sm_tile_size),
                idiv_ceil(K, reg_tile_size * sm_tile_size),
                1);
    dim3 threads(sm_tile_size, sm_tile_size, 1);
    // e.g. for 2d without register tiling
    //      blocks  = (PQ*N/8, K/8, 1)
    //      threads = (8, 8, 1)

    std::unique_ptr<gpu::primitive> conv(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {

            void** args_list = args.resolve_placeholder(0, &inputs[0])
                                   .resolve_placeholder(1, &inputs[1])
                                   .resolve_placeholder(2, &outputs[0])
                                   .get_argument_list();
            CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                          blocks.x,
                                          blocks.y,
                                          blocks.z,
                                          threads.x,
                                          threads.y,
                                          threads.z,
                                          0,
                                          NULL,
                                          args_list,
                                          0));
            debug_sync();
        }});

    return this->m_primitive_emitter->insert(std::move(conv));
}

void runtime::gpu::CUDAEmitter::print_tensor_from_gpu(codegen::CodeWriter& writer,
                                                      const std::string& tensor_name,
                                                      NVShape shape)
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

std::string runtime::gpu::CUDAEmitter::include_helpers()
{
    std::stringstream ss;
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
    ss << R"(
#define WARP_SIZE 32
#define __ballot_sync(mask, predicate) __ballot(predicate)
#define __shfl_down_sync(mask, val, delta, width) __shfl_down(val, delta, width)
#define __shfl_xor_sync(mask, val, laneMask, width) __shfl_xor(val, laneMask, width)
)";
#endif

    // add modern type definitions
    ss << "typedef signed char int8_t;\n";
    ss << "typedef signed short int16_t;\n";
    ss << "typedef signed int int32_t;\n";
    ss << "typedef signed long int int64_t;\n";
    ss << "typedef unsigned char uint8_t;\n";
    ss << "typedef unsigned short uint16_t;\n";
    ss << "typedef unsigned int uint32_t;\n";
    ss << "typedef unsigned long int uint64_t;\n";
    ss << "\n";

    // division_by_invariant_multiplication:
    // fast integer division via invariant multiplication and shifting
    // if value is a power of 2, magic will be 1 and only shifting
    // is required (predicate p below)
    // load: helper to load from constant memory for fast access
    ss << R"(
__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__device__ __forceinline__ void idiv_fast(int numerator, int denominator, float rcp,
                                          int& result, int& remainder)
{
    result = (int)((float)numerator * rcp);
    remainder = numerator - (result * denominator);
    result = (remainder >= denominator) ? (result + 1) : result;
    remainder = (remainder >= denominator) ? (remainder - denominator) : remainder;
}

__device__ __forceinline__ int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
__device__ __forceinline__ int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
__device__ __forceinline__ int msub16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, -%3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
)";
    return ss.str();
}

uint32_t runtime::gpu::CUDAEmitter::align_to_block_size(uint32_t threads, uint32_t block_size)
{
    if (threads > (1u << 31) - 1)
    {
        throw std::runtime_error("Cuda can't handle threads > 2^31 - 1.");
    }
    uint32_t r = (threads + block_size - 1) / block_size;
    return r;
}

void runtime::gpu::CUDAEmitter::sync()
{
    CUDA_SAFE_CALL(cuCtxSynchronize());
    return;
}

void runtime::gpu::CUDAEmitter::debug_sync()
{
#ifdef NGRAPH_DEBUG_ENABLE
    CUDA_SAFE_CALL(cuCtxSynchronize());
#endif
    return;
}
