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
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/cuda_emitter.hpp"
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
                                            const Shape& padding_interior,
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

    // if the kernel has not been compiled, build it
    auto compiled_kernel = ctx->compiled_kernel_pool->get(hash);
    if (compiled_kernel == nullptr)
    {
        // normalize pad dimensions to shape dimensions
        Shape pad_below(input_shape.size(), 0);
        Shape pad_above(input_shape.size(), 0);
        Shape pad_interior(input_shape.size(), 0);

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

        auto input_strides = row_major_strides(input_shape);
        auto output_strides = row_major_strides(output_shape);

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

        compiled_kernel = ctx->compiled_kernel_pool->set(hash, writer.get_code());
    }
    std::unique_ptr<gpu::primitive> pad;

    // if the pad value is statically provided, the kernel call signature is different
    if (pad_value == "") // pad value provided at runtime (dynamic)
    {
        pad.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
            void* args_list[] = {&inputs[1], &inputs[0], &outputs[0]};
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
        }});
    }
    else // pad value provided at compile time (static)
    {
        pad.reset(new gpu::primitive{[=](void** inputs, void** outputs) {
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
        }});
    }

    primitive_index = this->m_primitive_emitter->insert(std::move(pad));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDAEmitter::build_1d_max_pool(const GPURuntimeContext* ctx,
                                                    const std::array<std::string, 2>& dtypes,
                                                    const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    size_t window_width,
                                                    size_t window_stride)
{
    auto input_width = input_shape.back();
    auto output_width = output_shape.back();

    std::stringstream ss;
    ss << "maxpool"
       << "_i" << input_width << "_o" << output_width << "_w" << window_width << "_s"
       << window_stride;
    auto hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    auto nthreads = shape_size(output_shape);

    // if the kernel has not been compiled, build it
    auto compiled_kernel = ctx->compiled_kernel_pool->get(hash);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        // assumes data is in NCW format
        writer << "extern \"C\" __global__ void cuda_" << hash << "(" << dtypes[0] << "* in, "
               << dtypes[1] << "* out)\n";
        writer.block_begin();
        {
            // index into output tensor
            writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
            writer << "if (tid < " << nthreads << ")\n";
            writer.block_begin();
            {
                // index into input tensor
                writer << "size_t start = (tid / " << output_width << ") * " << input_width << " + "
                       << " (tid % " << output_width << ") * " << window_stride << ";\n";
                writer << dtypes[0] << " max_val = " << TypeInfo::Get(dtypes[0])->lowest() << ";\n";
                writer << "for (size_t i = start; i < start + " << window_width << "; i++)\n";
                writer.block_begin();
                {
                    writer << "const " << dtypes[0] << " input = in[i];\n";
                    writer << "if (input > max_val)\n";
                    writer.block_begin();
                    {
                        writer << "max_val = input;\n";
                    }
                    writer.block_end();
                }
                writer.block_end();
                writer << "out[tid] = max_val;\n";
            }
            writer.block_end();
        }
        writer.block_end();
        compiled_kernel = ctx->compiled_kernel_pool->set(hash, writer.get_code());
    }

    std::unique_ptr<gpu::primitive> pool(new gpu::primitive{[=](void** inputs, void** outputs) {
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
    }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pool));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

pooling_op_shape avgpool_shape(
    const Shape& in, const Shape& out, const Shape& window, const Shape& strides, const Shape& pad)
{
    pooling_op_shape shape;
    shape.N = static_cast<int>(in[0]);
    shape.C = static_cast<int>(in[1]);
    shape.K = shape.C; // pooling feature maps is
    shape.J = shape.C; // not currently supported
    if (in.size() == 3)
    {
        shape.D = 1;
        shape.H = 1;
        shape.W = static_cast<int>(in[2]);
        shape.M = 1;
        shape.P = 1;
        shape.Q = static_cast<int>(out[2]);
        shape.T = 1;
        shape.R = 1;
        shape.S = static_cast<int>(window[0]);
        shape.STRIDE_D = 0;
        shape.STRIDE_H = 0;
        shape.STRIDE_W = static_cast<int>(strides[0]);
        shape.PAD_D = 0;
        shape.PAD_H = 0;
        shape.PAD_W = static_cast<int>(pad[0]);
    }
    else if (in.size() == 4)
    {
        shape.D = 1;
        shape.H = static_cast<int>(in[2]);
        shape.W = static_cast<int>(in[3]);
        shape.M = 1;
        shape.P = static_cast<int>(out[2]);
        shape.Q = static_cast<int>(out[3]);
        shape.T = 1;
        shape.R = static_cast<int>(window[0]);
        shape.S = static_cast<int>(window[1]);
        shape.STRIDE_D = 0;
        shape.STRIDE_H = static_cast<int>(strides[0]);
        shape.STRIDE_W = static_cast<int>(strides[1]);
        shape.PAD_D = 0;
        shape.PAD_H = static_cast<int>(pad[0]);
        shape.PAD_W = static_cast<int>(pad[1]);
    }
    else if (in.size() == 5)
    {
        shape.D = static_cast<int>(in[2]);
        shape.H = static_cast<int>(in[3]);
        shape.W = static_cast<int>(in[4]);
        shape.M = static_cast<int>(out[2]);
        shape.P = static_cast<int>(out[3]);
        shape.Q = static_cast<int>(out[4]);
        shape.T = static_cast<int>(window[0]);
        shape.R = static_cast<int>(window[1]);
        shape.S = static_cast<int>(window[2]);
        shape.STRIDE_D = static_cast<int>(strides[0]);
        shape.STRIDE_H = static_cast<int>(strides[1]);
        shape.STRIDE_W = static_cast<int>(strides[2]);
        shape.PAD_D = static_cast<int>(pad[0]);
        shape.PAD_H = static_cast<int>(pad[1]);
        shape.PAD_W = static_cast<int>(pad[2]);
    }
    else
    {
        throw std::runtime_error("AvgPool currently supports up to 3 spatial dimensions.");
    }
    return shape;
}

size_t runtime::gpu::CUDAEmitter::build_avg_pool(const GPURuntimeContext* ctx,
                                                 const std::array<std::string, 2>& dtypes,
                                                 const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 const Shape& window_shape,
                                                 const Shape& window_stride,
                                                 const Shape& padding_below,
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
    auto compiled_kernel = ctx->compiled_kernel_pool->get(kernel_name);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        writer << include_helpers();
        // In the pooling operation out = P(in) where in: NCDHW -> out: NKMPQ
        // via pooling window: JTRS. Currently feature pooling
        // is not supported and so K = C and J is unused
        writer << "extern \"C\" __global__ void cuda_" << kernel_name << "(" << dtypes[0]
               << "* in, " << dtypes[1] << "* out, "
               << "float alpha, float beta, "
               << "int N, int C, int D, int H, int W, "
               << "int HW, int DHW, int CDHW, int magic_N, int shift_N, "
               << "int P, int Q, int magic_P, int shift_P, "
               << "int PQ, int MPQ, int KMPQ, "
               << "int S, int RS, int TRS, "
               << "int magic_S, int shift_S, int magic_RS, int shift_RS, "
               << "int str_d, int str_h, int str_w, "
               << "int pad_d, int pad_h, int pad_w"
               << ")\n";
        writer.block_begin();
        {
            writer << "const int tid = threadIdx.x;\n";
            writer << "if (tid < 32)\n";
            writer.block_begin();
            {
                writer << "const int q = blockIdx.x;\n";
                writer << "const int mp = blockIdx.y;\n";
                writer << "const int nk = blockIdx.z;\n";
                writer << "const int k = div64(nk, magic_N, shift_N);\n";
                writer << "const int n = nk - k * N;\n";
                writer << "const int m = div64(mp, magic_P, shift_P);\n";
                writer << "const int p = mp - m * P;\n";
                writer << "out += n*KMPQ + k*MPQ + m*PQ + mad16(p, Q, q);\n";

                // coordinate transform factors from MPQ to DHW
                writer << "int qs = q * str_w - pad_w;\n";
                writer << "int pr = p * str_h - pad_h;\n";
                writer << "int mt = m * str_d - pad_d;\n";

                writer << "int pool_size = ";
                auto pool_size = include_pad ? "TRS" : "0";
                writer << pool_size << ";\n";

                writer << "float sum = 0.0f;\n";
                writer << "float rcp_pool_size = 1.0f;\n";
                // each warp operates on a single pooling window and
                // reduces the contents of the window within the warp
                writer << "for (int trs = tid; trs < TRS; trs += 32)\n";
                writer.block_begin();
                {
                    writer << "int t = div64(trs, magic_RS, shift_RS);\n";
                    writer << "int rs = mod16(trs, t, RS);\n";
                    writer << "int r  = div64(rs, magic_S, shift_S);\n";
                    writer << "int s  = mod16(rs, r, S);\n";

                    // coordinate transformation from TRS to DHW
                    // via MPQ transform factors above
                    writer << "int x = qs + s;\n";
                    writer << "int y = pr + r;\n";
                    writer << "int z = mt + t;\n";

                    // helper to check participating threads
                    writer << "bool bounds_x = (x >= 0) && (x < W);\n";
                    writer << "bool bounds_y = (y >= 0) && (y < H);\n";
                    writer << "bool bounds_z = (z >= 0) && (z < D);\n";
                    writer << "bool within_tensor_bounds = bounds_x && bounds_y && bounds_z;\n";

                    if (include_pad == false)
                    {
                        // count the number of (non-padded) elements
                        writer << "pool_size += __popc(__ballot(within_tensor_bounds));\n";
                    }
                    // this will need to change to k->c once
                    // feature pooling support is added
                    writer << "int idx = n*CDHW + k*DHW + z*HW + y*W + x;\n";
                    writer << "sum += load(in,idx,within_tensor_bounds);\n";
                }
                writer.block_end();

                writer << "rcp_pool_size = 1.0f / (float)pool_size;\n";
                // reduce pooling window within warp.
                // this could be improved by calculating the
                // pooling windows each thread can partake in to
                // reduce loads and increase coalescing. in that case,
                // multiple warps per block would be required and the
                // warp reduced sums would need to be accumulated in
                // shared memory
                writer << "for (int i = 16; i > 0; i >>= 1)\n";
                writer.block_begin();
                {
                    writer << "sum += __shfl_xor(sum,i);\n";
                }
                writer.block_end();
                // write result to output
                writer << "if (tid == 0)\n";
                writer.block_begin();
                {
                    writer << "*out = sum * rcp_pool_size;\n";
                }
                writer.block_end();
            }
            writer.block_end();
        }
        writer.block_end();
        compiled_kernel = ctx->compiled_kernel_pool->set(kernel_name, writer.get_code());
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
            CUDA_SAFE_CALL(cuCtxSynchronize());

        }});

    primitive_index = this->m_primitive_emitter->insert(std::move(pool));
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

void runtime::gpu::CUDAEmitter::print_tensor_from_gpu(codegen::CodeWriter& writer,
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

std::string runtime::gpu::CUDAEmitter::include_helpers()
{
    // div64: fast integer division via magic multiplication and shifting
    // if value is a power of 2, magic will be 1 and only shifting
    // is required (predicate p in div64)
    // load: helper to load from constant memory for fast access
    std::stringstream ss;
    ss << R"(
__device__ __forceinline__ int div64(int value, int magic, int shift)
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
)";
    return ss.str();
}
