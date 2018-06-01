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
                                                         const std::vector<std::string>& dtypes)
{
    auto num_inputs = dtypes.size() - 1;
    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    for (size_t i = 0; i < num_inputs; i++)
    {
        writer << dtypes[i] << "* in" << i << ", ";
    }
    writer << dtypes[num_inputs] << "* out, "
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
                                                       const std::array<std::string, 2>& dtypes)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1] << "* out, size_t m, size_t k, size_t n)\n";
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
                                                    const std::array<std::string, 2>& dtypes)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1] << "* out, size_t m, size_t k, size_t n)\n";
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
                                                     const std::array<std::string, 2>& dtypes)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1]
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
                                                    const std::vector<std::string>& dtypes,
                                                    size_t num_inputs)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    for (size_t i = 0; i < num_inputs; i++)
    {
        writer << dtypes[i] << "* in" << i << ", ";
    }
    writer << dtypes[num_inputs]
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
                                                   const std::array<std::string, 2>& dtypes)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1] << "* out, size_t* input_strides, size_t* lower_bounds, size_t* "
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
                                                     const std::array<std::string, 2>& dtypes)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1]
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

void runtime::gpu::CudaKernelBuilder::get_replace_slice_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 3>& dtypes,
    int nthreads_per_block)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << dtypes[0] << "* in, "
           << dtypes[1] << "* source, " << dtypes[2] << "* out, "
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
           << "int rank, "
           << "int nthreads"
           << ")\n";
    writer.block_begin();
    {
        writer << "extern __shared__ int coordinate[];\n";
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
                writer << "coordinate[i] = division_by_invariant_multiplication(dim_product, "
                          "dim_magic[data_idx], "
                          "dim_shift[data_idx]);\n";
                writer << "dim_product -= (coordinate[i] * dim_strides[data_idx]);\n";
                writer << "data_idx++;\n";
            }
            writer.block_end();
            writer << "coordinate[threadIdx.x + (rank-1) * " << nthreads_per_block
                   << "] = dim_product;\n";
            writer << "data_idx = 0;\n";
            writer << "bool in_bounds = true;\n";
            writer << "int source_idx = 0;\n";
            writer << "for (int i = threadIdx.x; i < rank * " << nthreads_per_block
                   << "; i += " << nthreads_per_block << ")\n";
            writer.block_begin();
            {
                writer << "int source_di = division_by_invariant_multiplication(coordinate[i], "
                          "slice_magic[data_idx], "
                          "slice_shift[data_idx]);\n";
                writer << "bool on_stride = (mod16(coordinate[i], source_di, "
                          "slice_str[data_idx]) == 0);\n";
                // within slice of input tensor and a multiple of the slice stride
                writer << "bool in_slice_di = (coordinate[i] >= lower_bounds[data_idx]) && "
                          "(coordinate[i] < upper_bounds[data_idx]) && on_stride;\n";
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

void runtime::gpu::CudaKernelBuilder::get_softmax_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 2>& dtypes,
    const int rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "("
               << dtypes[0] << "* in, "
               << dtypes[1] << "* out, "
               << "int* strides, "
               << "int* stride_magic, "
               << "int* stride_shift, "
               << "int* reduced_strides, "
               << "float alpha, float beta, "
               << "size_t nthreads"
               << ")\n";
    writer.block_begin();
    {
        writer << "unsigned start = clock();\n";

        writer << "const int tid = blockDim.x*blockIdx.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            // calculate tensor coordinates
            writer << "int dim_product = tid;\n";
            for (int i = 0; i < rank; i++)
            {
                writer << "int coordinate" << i << " = division_by_invariant_multiplication("
                       << "dim_product, stride_magic[" << i << "], stride_shift[" << i << "]);\n";
                writer << "dim_product -= (coordinate" << i << " * strides[" << i << "]);\n";
            }
            writer << "int reduced_idx = 0;\n";
            for (int i = 0; i < rank; i++)
            {
                writer << "reduced_idx += coordinate" << i << " * reduced_strides[" << i << "];\n";

            }





            // TODO: mediate atomic memory access contention
            writer << dtypes[0] << " val = expf(load(in, tid));\n";
            writer << "atomicAdd(&out[reduced_idx], val);\n";
            writer << "__threadfence();\n";
            writer << dtypes[1] << " sum = out[reduced_idx];\n";
            writer << "out[tid] = val/sum;\n";

            writer << R"(
        unsigned end = clock();
        unsigned time;
        if (end > start)
                time = end - start;
        else
                time = end + (0xffffffff - start);
        printf("tid = %d, time = %d\n",tid,time);
)";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_device_helper(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& math_kernel,
                                                        const std::vector<std::string>& dtypes)
{
    if (math_kernel.size())
    {
        auto num_inputs = dtypes.size() - 1;
        writer << "__device__ __forceinline__ " << dtypes[num_inputs] << " " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; i++)
        {
            writer << dtypes[i] << " x" << i << ", ";
        }
        writer << dtypes[num_inputs - 1] << " x" << num_inputs - 1;
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
