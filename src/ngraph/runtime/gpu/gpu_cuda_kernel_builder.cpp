/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"

using namespace ngraph;

void runtime::gpu::CudaKernelBuilder::get_elementwise_op(codegen::CodeWriter& writer,
                                                         const std::string& name,
                                                         const std::string& op,
                                                         const std::vector<std::string>& data_types,
                                                         const size_t& num_inputs)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    for (size_t i = 0; i < num_inputs; i++)
    {
        writer << data_types[i] << "* in" << i << ", ";
    }
    writer << data_types[num_inputs] << "* out, "
           << "size_t n)\n";
    writer << "{\n";
    writer.indent++;
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
        writer << "if (tid < n)\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "out[tid] = " << op << "(";
            for (size_t i = 0; i < num_inputs - 1; i++)
            {
                writer << "in" << i << "[tid], ";
            }
            writer << "in" << num_inputs - 1 << "[tid]);\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    writer.indent--;
    writer << "}\n";

    return;
}

void runtime::gpu::CudaKernelBuilder::get_broadcast_op(codegen::CodeWriter& writer,
                                                       const std::string& name,
                                                       const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, size_t m, size_t k, size_t n)\n";
    writer << "{\n";
    writer.indent++;
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "size_t idx = tid / (m * k) * m + tid % m;\n";
            writer << "out[tid] = in[idx];\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::CudaKernelBuilder::get_onehot_op(codegen::CodeWriter& writer,
                                                    const std::string& name,
                                                    const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, size_t m, size_t k, size_t n)\n";
    writer << "{\n";
    writer.indent++;
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "size_t idx = (tid / m) * m * k + (m * in[tid]) + tid % m;\n";
            writer << "out[idx] = 1;\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::CudaKernelBuilder::get_reshape_op(codegen::CodeWriter& writer,
                                                     const std::string& name,
                                                     const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1]
           << "* out, size_t* input_strides, size_t* trans_strides, size_t rank, size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "size_t idx_in = tid;\n";
            writer << "size_t idx_out = 0;\n";

            writer << "for(size_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "idx_out += (idx_in / input_strides[i]) * trans_strides[i];\n";
                writer << "idx_in %= input_strides[i];\n";
            }
            writer.block_end();
            writer << "out[idx_out] = in[tid];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_concat_op(codegen::CodeWriter& writer,
                                                    const std::string& name,
                                                    const std::vector<std::string>& data_types,
                                                    size_t num_inputs)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    for (size_t i = 0; i < num_inputs; i++)
    {
        writer << data_types[i] << "* in" << i << ", ";
    }
    writer << data_types[num_inputs]
           << "* out, size_t* block_strides, size_t block_size, size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if(tid < n)\n";
        writer.block_begin();
        {
            writer << "out[tid] = 1;\n";
            writer << "size_t idx_out = tid;\n";
            writer << "size_t block_id = tid / block_size;\n";
            writer << "size_t block_idx = tid % block_size;\n";
            writer << "bool processed = false;\n";
            for (size_t i = 0; i < num_inputs; i++)
            {
                writer << "if(!processed && (block_idx < block_strides[" << i << "]))\n";
                writer.block_begin();
                {
                    writer << "out[idx_out] = in" << i << "[block_id * block_strides[" << i
                           << "] + block_idx];";
                    writer << "processed = true;\n";
                }
                writer.block_end();
                writer << "block_idx -= block_strides[" << i << "];\n";
            }
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_slice_op(codegen::CodeWriter& writer,
                                                   const std::string& name,
                                                   const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, size_t* input_strides, size_t* lower_bounds, size_t* "
                               "slice_strides, size_t* output_strides, size_t rank, size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "size_t idx_in = 0;\n";
            writer << "size_t idx_out = tid;\n";

            writer << "for(size_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "idx_in += (((idx_out / output_strides[i]) * slice_strides[i]) + "
                          "lower_bounds[i]) * input_strides[i];\n";
                writer << "idx_out %= output_strides[i];\n";
            }
            writer.block_end();
            writer << "out[tid] = in[idx_in];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_reverse_op(codegen::CodeWriter& writer,
                                                     const std::string& name,
                                                     const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1]
           << "* out, size_t* input_shape, size_t* reverse_axes, size_t rank, size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "size_t idx_in = tid;\n";
            writer << "size_t idx_out = 0;\n";
            writer << "size_t stride = 1;\n";
            writer << "for(size_t i = rank; i > 0; i--)\n";
            writer.block_begin();
            {
                writer << "size_t idx = i - 1;\n";
                writer << "size_t axes_i_in = idx_in % input_shape[idx];\n";
                writer << "idx_in /= input_shape[idx];\n";
                writer << "size_t axes_i_out = reverse_axes[idx] ? input_shape[idx] - axes_i_in - "
                          "1 : axes_i_in;\n";
                writer << "idx_out += axes_i_out * stride;\n";
                writer << "stride *= input_shape[idx];\n";
            }
            writer.block_end();
            writer << "out[idx_out] = in[tid];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_device_helper(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& math_kernel,
                                                        const std::vector<std::string>& data_types,
                                                        const size_t& num_inputs)
{
    if (math_kernel.size())
    {
        writer << "__device__ " << data_types[num_inputs] << " " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; i++)
        {
            writer << data_types[i] << " x" << i << ", ";
        }
        writer << data_types[num_inputs - 1] << " x" << num_inputs - 1;
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "return " + math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return;
}

void runtime::gpu::CudaKernelBuilder::add_pod_typedefs(codegen::CodeWriter& writer)
{
    writer << "typedef signed char int8_t;\n";
    writer << "typedef signed short int16_t;\n";
    writer << "typedef signed int int32_t;\n";
    writer << "typedef signed long int int64_t;\n";
    writer << "typedef unsigned char uint8_t;\n";
    writer << "typedef unsigned short uint16_t;\n";
    writer << "typedef unsigned int uint32_t;\n";
    writer << "typedef unsigned long int uint64_t;\n";
    writer << "\n";
}
