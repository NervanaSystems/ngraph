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
                                                         const std::vector<std::string>& data_types)
{
    auto num_inputs = data_types.size() - 1;
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
            writer << "size_t input_idx = tid;\n";
            writer << "size_t output_idx = 0;\n";

            writer << "for(size_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "output_idx += (input_idx / input_strides[i]) * trans_strides[i];\n";
                writer << "input_idx %= input_strides[i];\n";
            }
            writer.block_end();
            writer << "out[output_idx] = in[tid];\n";
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
            writer << "size_t output_idx = tid;\n";
            writer << "size_t block_id = tid / block_size;\n";
            writer << "size_t block_idx = tid % block_size;\n";
            writer << "bool processed = false;\n";
            for (size_t i = 0; i < num_inputs; i++)
            {
                writer << "if(!processed && (block_idx < block_strides[" << i << "]))\n";
                writer.block_begin();
                {
                    writer << "out[output_idx] = in" << i << "[block_id * block_strides[" << i
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


void runtime::gpu::CudaKernelBuilder::get_pad_dilation_op(codegen::CodeWriter& writer,
                                                   const std::string& name,
                                                   const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, size_t* input_strides, size_t* output_strides, size_t* padding_below, size_t* "
                               "dilation_strides, size_t rank, size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "size_t output_idx = 0;\n";
            writer << "size_t input_idx = tid;\n";

            writer << "for(size_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "output_idx += (((input_idx / input_strides[i] + padding_below[i]) * dilation_strides[i])"
                          "* output_strides[i];\n";
                writer << "input_idx %= input_strides[i];\n";
            }
            writer.block_end();
            writer << "out[output_idx] = in[tid];\n";
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
            writer << "size_t input_idx = 0;\n";
            writer << "size_t output_idx = tid;\n";

            writer << "for(size_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "input_idx += (((output_idx / output_strides[i]) * slice_strides[i]) + "
                          "lower_bounds[i]) * input_strides[i];\n";
                writer << "output_idx %= output_strides[i];\n";
            }
            writer.block_end();
            writer << "out[tid] = in[input_idx];\n";
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
            writer << "size_t input_idx = tid;\n";
            writer << "size_t output_idx = 0;\n";
            writer << "size_t stride = 1;\n";
            writer << "for(size_t i = rank; i > 0; i--)\n";
            writer.block_begin();
            {
                writer << "size_t idx = i - 1;\n";
                writer << "size_t axes_i_in = input_idx % input_shape[idx];\n";
                writer << "input_idx /= input_shape[idx];\n";
                writer << "size_t axes_i_out = reverse_axes[idx] ? input_shape[idx] - axes_i_in - "
                          "1 : axes_i_in;\n";
                writer << "output_idx += axes_i_out * stride;\n";
                writer << "stride *= input_shape[idx];\n";
            }
            writer.block_end();
            writer << "out[output_idx] = in[tid];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_reduce_window_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::string& op,
    const std::vector<std::string>& data_types,
    const size_t rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, int* input_strides, int* output_shape, int* "
                               "reduce_window_shape, int* reduce_window_strides, size_t n)\n";
    writer.block_begin();
    {
        writer << "const int tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "int output_idx = tid;\n";
            writer << "int idx_init = 0; //result will be initial to in[idx_init]\n";
            for (int i = static_cast<int>(rank) - 1; i >= 0; i--)
            {
                writer << "int output_idx_" << i << " = output_idx % output_shape[" << i << "];\n";
                writer << "int input_idx_start_for_axis_" << i << " = output_idx_" << i
                       << " * reduce_window_strides[" << i << "];\n";
                writer << "int input_idx_end_for_axis_" << i << " = input_idx_start_for_axis_" << i
                       << " + reduce_window_shape[" << i << "];\n";
                writer << "idx_init += input_idx_start_for_axis_" << i << " * input_strides[" << i
                       << "];\n";
                writer << "output_idx /= output_shape[" << i << "];\n";
            }

            writer << data_types[1] << " result = in[idx_init];\n";

            for (int i = 0; i < rank; i++)
            {
                writer << "for(int i_" << i << " = input_idx_start_for_axis_" << i << "; i_" << i
                       << " < input_idx_end_for_axis_" << i << "; i_" << i << "++)\n";
                writer.block_begin();
            }

            writer << "int input_idx = 0;\n";
            for (int i = 0; i < rank; i++)
            {
                writer << "input_idx += i_" << i << " * input_strides[" << i << "];\n";
            }
            writer << "result = (input_idx == idx_init) ? result : " << op
                   << "(result, in[input_idx]); //skip in[idx_init] in loop\n";
            for (int i = 0; i < rank; i++)
            {
                writer.block_end();
            }
            writer << "out[tid] = result;\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_replace_slice_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 3>& data_types,
    int nthreads_per_block)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* source, " << data_types[2] << "* out, "
           << "float alpha, float beta, "
           << "int* dim_strides, "
           << "int* dim_magic, "
           << "int* dim_shift, "
           << "int* lower_bounds, "
           << "int* upper_bounds, "
           << "int* slice_str, "
           << "int* slice_magic, "
           << "int* slice_shift, "
           << "int* dim_source, "
           << "int* src_strides, "
           << "int rank,"
           << "size_t nthreads"
           << ")\n";
    writer.block_begin();
    {
        writer << "extern __shared__ int dimensions[];\n";
        writer << "const int tid = blockDim.x*blockIdx.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            writer << "int dim_product = tid;\n";
            writer << "int data_idx = 0;\n";
            writer << "for (int i = threadIdx.x; i < (rank - 1) * " << nthreads_per_block
                   << "; i += " << nthreads_per_block << ")\n";
            writer.block_begin();
            {
                writer << "dimensions[i] = division_by_invariant_multiplication(dim_product, "
                          "dim_magic[data_idx], "
                          "dim_shift[data_idx]);\n";
                writer << "dim_product -= (dimensions[i] * dim_strides[data_idx]);\n";
                writer << "data_idx++;\n";
            }
            writer.block_end();
            writer << "dimensions[threadIdx.x + (rank-1) * " << nthreads_per_block
                   << "] = dim_product;\n";
            writer << "data_idx = 0;\n";
            writer << "bool in_bounds = true;\n";
            writer << "int source_idx = 0;\n";
            writer << "for (int i = threadIdx.x; i < rank * " << nthreads_per_block
                   << "; i += " << nthreads_per_block << ")\n";
            writer.block_begin();
            {
                writer << "int source_di = division_by_invariant_multiplication(dimensions[i], "
                          "slice_magic[data_idx], "
                          "slice_shift[data_idx]);\n";
                writer << "bool on_stride = (mod16(dimensions[i], source_di, "
                          "slice_str[data_idx]) == 0);\n";
                // within slice of input tensor and a multiple of the slice stride
                writer << "bool in_slice_di = (dimensions[i] >= lower_bounds[data_idx]) && "
                          "(dimensions[i] < upper_bounds[data_idx]) && on_stride;\n";
                writer << "in_bounds = in_bounds && in_slice_di;\n";
                // subtract off lower bound to convert to source index
                writer << "source_di -= lower_bounds[data_idx];\n";
                writer << "source_idx += source_di * src_strides[data_idx];\n";
                writer << "data_idx++;\n";
            }
            writer.block_end();
            writer << "out[tid] = in_bounds ? source[source_idx] : in[tid];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_device_helper(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& math_kernel,
                                                        const std::vector<std::string>& data_types)
{
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "__device__ __forceinline__ " << data_types[num_inputs] << " " << name << "(";
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
