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
#include "ngraph/runtime/gpu/type_info.hpp"

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
           << "uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
        writer << "uint32_t step = gridDim.x * blockDim.x; \n";
        writer << "for ( ;tid < n; tid += step)\n";
        writer.block_begin();
        {
            writer << "out[tid] = " << op << "(";
            for (size_t i = 0; i < num_inputs - 1; i++)
            {
                writer << "in" << i << "[tid], ";
            }
            writer << "in" << num_inputs - 1 << "[tid]);\n";
        }
        writer.block_end();
    }
    writer.block_end();

    return;
}

void runtime::gpu::CudaKernelBuilder::get_ew_collective_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::string& op,
    const std::string& reduce_op,
    const std::vector<std::string>& data_types,
    const std::set<size_t>& reduced_tensors,
    bool save_elementwise,
    size_t rank)
{
    auto num_inputs = data_types.size() - 1;
    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    for (size_t i = 0; i < num_inputs; i++)
    {
        writer << data_types[i] << "* in" << i << ", ";
    }
    writer << data_types[num_inputs] << "* out0, ";

    // multi-output to save intermediate elementwise op if requested
    if (save_elementwise)
    {
        writer << data_types[num_inputs] << "* out1, ";
    }
    writer << "int* strides, "
           << "int* stride_magic, "
           << "int* stride_shift, "
           << "int* reduced_strides, "
           << "size_t n)\n";
    writer.block_begin();
    {
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            std::string reduced_idx = collective_coordinate_transform_helper(writer,
                                                                             "tid",
                                                                             "strides",
                                                                             "stride_magic",
                                                                             "stride_shift",
                                                                             "reduced_strides",
                                                                             "coordinate",
                                                                             rank);
            // element-wise operation
            writer << data_types[num_inputs] << " output = " << op << "(";
            for (size_t i = 0; i < num_inputs; i++)
            {
                if (i > 0)
                {
                    writer << ", ";
                }
                writer << "in" << i << "[";
                if (reduced_tensors.count(i) > 0)
                {
                    writer << reduced_idx;
                }
                else
                {
                    writer << "tid";
                }
                writer << "]";
            }
            writer << ");\n";

            // global collective reduce or broadcast
            if (reduce_op != "")
            {
                // TODO: mediate atomic memory access contention
                writer << reduce_op << "(&out0[" << reduced_idx << "], output);\n";
                if (save_elementwise)
                {
                    writer << "out1["
                           << "tid"
                           << "] = output;\n";
                }
            }
            else
            {
                writer << "out0[tid] = output;\n";
                if (save_elementwise)
                {
                    writer << "out1[" << reduced_idx << "] = output;\n";
                }
            }
        }
        writer.block_end();
    }
    writer.block_end();

    return;
}

void runtime::gpu::CudaKernelBuilder::get_broadcast_op(codegen::CodeWriter& writer,
                                                       const std::string& name,
                                                       const std::array<std::string, 2>& data_types,
                                                       const size_t rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, "
           << "int* strides, "
           << "int* stride_magic, "
           << "int* stride_shift, "
           << "int* reduced_strides, "
           << "float alpha, float beta, "
           << "size_t nthreads"
           << ")\n";
    writer.block_begin();
    {
        writer << "const int tid = blockDim.x*blockIdx.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            // calculate tensor coordinates (inverse tensor reduction)
            std::string reduced_idx = collective_coordinate_transform_helper(writer,
                                                                             "tid",
                                                                             "strides",
                                                                             "stride_magic",
                                                                             "stride_shift",
                                                                             "reduced_strides",
                                                                             "coordinate",
                                                                             rank);
            writer << "out[tid] = load(in, " << reduced_idx << ");\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_onehot_op(codegen::CodeWriter& writer,
                                                    const std::string& name,
                                                    const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, uint32_t m, uint32_t k, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t idx = (tid / m) * m * k + (m * in[tid]) + tid % m;\n";
            writer << "out[idx] = 1;\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::get_reshape_op(codegen::CodeWriter& writer,
                                                     const std::string& name,
                                                     const std::array<std::string, 2>& data_types,
                                                     size_t rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1]
           << "* out, uint32_t* input_strides, uint32_t* trans_strides, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t input_idx = tid;\n";
            writer << "uint32_t output_idx = 0;\n";
            size_t i = 0;
            for (; i < rank - 1; i++)
            {
                writer << "output_idx += (input_idx / input_strides[" << i << "]) * trans_strides["
                       << i << "];\n";
                writer << "input_idx %= input_strides[" << i << "];\n";
            }
            writer << "output_idx += (input_idx / input_strides[" << i << "]) * trans_strides[" << i
                   << "];\n";
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
           << "* out, uint32_t* block_strides, uint32_t block_size, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if(tid < n)\n";
        writer.block_begin();
        {
            writer << "out[tid] = 1;\n";
            writer << "uint32_t output_idx = tid;\n";
            writer << "uint32_t block_id = tid / block_size;\n";
            writer << "uint32_t block_idx = tid % block_size;\n";
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

void runtime::gpu::CudaKernelBuilder::get_pad_dynamic_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 2>& data_types)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, uint32_t* input_strides, uint32_t* output_strides, "
                               "uint32_t* padding_below, uint32_t* "
                               "padding_interior, uint32_t rank, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t output_idx = 0;\n";
            writer << "uint32_t input_idx = tid;\n";

            writer << "for(uint32_t i = 0; i < rank; i++)\n";
            writer.block_begin();
            {
                writer << "output_idx += (input_idx / input_strides[i] * padding_interior[i]  + "
                          "padding_below[i]) "
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

void runtime::gpu::CudaKernelBuilder::get_reverse_sequence_op(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 3>& data_types,
    const size_t batch_axis,
    const size_t sequence_axis,
    const size_t rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* sequence, " << data_types[2] << "* out, "
           << "uint32_t* output_shape, uint32_t* output_strides, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t input_idx = tid;\n";
            for (size_t i = 0; i < rank - 1; i++)
            {
                writer << "uint32_t output_idx_" << i << " = input_idx / output_strides[" << i
                       << "];\n";
                writer << "input_idx %= output_strides[" << i << "];\n";
            }
            writer << "uint32_t output_idx_" << rank - 1 << " = input_idx / output_strides["
                   << rank - 1 << "];\n";
            writer << "uint32_t sequence_length = sequence[output_idx_" << batch_axis << "];\n";
            writer << "assert(sequence_length <= output_shape[" << sequence_axis << "]);\n";

            writer << "bool need_reverse = (output_idx_" << sequence_axis
                   << " < sequence_length) && (sequence_length > 1);\n";
            writer << "output_idx_" << sequence_axis
                   << " = need_reverse ? sequence_length - output_idx_" << sequence_axis
                   << " - 1 : output_idx_" << sequence_axis << ";\n";
            writer << "uint32_t output_idx = need_reverse ? ";
            writer << "output_idx_" << 0 << " * output_strides[" << 0 << "]";
            for (size_t i = 1; i < rank; i++)
            {
                writer << " + output_idx_" << i << " * output_strides[" << i << "]";
            }
            writer << " : tid;\n";
            writer << "out[output_idx] = in[tid];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}
void runtime::gpu::CudaKernelBuilder::get_slice_op(codegen::CodeWriter& writer,
                                                   const std::string& name,
                                                   const std::array<std::string, 2>& data_types,
                                                   size_t rank)
{
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, uint32_t* input_strides, uint32_t* lower_bounds, uint32_t* "
                               "slice_strides, uint32_t* output_strides, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t input_idx = 0;\n";
            writer << "uint32_t output_idx = tid;\n";
            size_t i = 0;
            for (; i < rank - 1; i++)
            {
                writer << "input_idx += (((output_idx / output_strides[" << i
                       << "]) * slice_strides[" << i << "]) + "
                                                        "lower_bounds["
                       << i << "]) * input_strides[" << i << "];\n";
                writer << "output_idx %= output_strides[" << i << "];\n";
            }
            writer << "input_idx += (((output_idx / output_strides[" << i << "]) * slice_strides["
                   << i << "]) + "
                           "lower_bounds["
                   << i << "]) * input_strides[" << i << "];\n";
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
           << "* out, uint32_t* input_shape, uint32_t* reverse_axes, uint32_t rank, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t input_idx = tid;\n";
            writer << "uint32_t output_idx = 0;\n";
            writer << "uint32_t stride = 1;\n";
            writer << "for(uint32_t i = rank; i > 0; i--)\n";
            writer.block_begin();
            {
                writer << "uint32_t idx = i - 1;\n";
                writer << "uint32_t axes_i_in = input_idx % input_shape[idx];\n";
                writer << "input_idx /= input_shape[idx];\n";
                writer
                    << "uint32_t axes_i_out = reverse_axes[idx] ? input_shape[idx] - axes_i_in - "
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

void runtime::gpu::CudaKernelBuilder::get_max_pool_1d(codegen::CodeWriter& writer,
                                                      const std::string& name,
                                                      const std::array<std::string, 2>& data_types,
                                                      size_t input_width,
                                                      size_t output_width,
                                                      size_t window_width,
                                                      size_t window_stride)
{
    // assumes data is in NCW format
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, size_t nthreads)\n";
    writer.block_begin();
    {
        // index into output tensor
        writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            // index into input tensor
            writer << "size_t start = (tid / " << output_width << ") * " << input_width << " + "
                   << " (tid % " << output_width << ") * " << window_stride << ";\n";
            writer << data_types[0] << " max_val = " << TypeInfo::Get(data_types[0])->lowest()
                   << ";\n";
            writer << "for (size_t i = start; i < start + " << window_width << "; i++)\n";
            writer.block_begin();
            {
                writer << "const " << data_types[0] << " input = in[i];\n";
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
}

void runtime::gpu::CudaKernelBuilder::get_avg_pool(codegen::CodeWriter& writer,
                                                   const std::string& name,
                                                   const std::array<std::string, 2>& data_types,
                                                   bool include_pad)
{
    // In the pooling operation out = P(in) where in: NCDHW -> out: NKMPQ
    // via pooling window: JTRS. Currently feature pooling
    // is not supported and so K = C and J is unused
    writer << "extern \"C\" __global__ void cuda_" << name << "(" << data_types[0] << "* in, "
           << data_types[1] << "* out, "
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
            writer << "const int k = division_by_invariant_multiplication(nk, magic_N, "
                      "shift_N);\n";
            writer << "const int n = nk - k * N;\n";
            writer << "const int m = division_by_invariant_multiplication(mp, magic_P, "
                      "shift_P);\n";
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
                writer << "int t = division_by_invariant_multiplication(trs, magic_RS, "
                          "shift_RS);\n";
                writer << "int rs = mod16(trs, t, RS);\n";
                writer << "int r  = division_by_invariant_multiplication(rs, magic_S, shift_S);\n";
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
                    writer << "pool_size += __popc(__ballot_sync(0xffffffff, "
                              "within_tensor_bounds));\n";
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
                writer << "sum += __shfl_xor_sync(0xffffffff,sum,i,32);\n";
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
}

void runtime::gpu::CudaKernelBuilder::get_convolution_forward(
    codegen::CodeWriter& writer,
    const std::string& name,
    const std::array<std::string, 3>& data_types,
    int N,
    int K,
    int filter_size,
    int rank,
    int sm_tile_size,
    int reg_tile_size)
{
    writer << "#define NUM_ROWS 8\n";
    writer << "#define FILTER_SIZE " << filter_size << "\n";
    writer << "#define SM_TILE_SIZE " << sm_tile_size << "\n";
    writer << "#define REG_TILE_SIZE " << reg_tile_size << "\n";
    // convenient type def for register tiling
    writer << "typedef union Matrix\n";
    writer.block_begin();
    {
        writer << data_types[0] << reg_tile_size << " f" << reg_tile_size << ";\n";
        writer << data_types[0] << " f[" << reg_tile_size << "];\n";
    }
    writer.block_end();
    writer << "Matrix;\n\n";

    writer << "extern \"C\" __global__ void cuda_" << name << "(";
    writer << data_types[0] << "* in, ";
    writer << data_types[1] << "* filter, ";
    writer << data_types[2] << "* out, ";
    // TODO: add alpha/beta support
    writer << "float alpha, float beta, "
           << "int N, "
           << "int C, "
           << "int K, "
           << "int input_channel_size, "
           << "int filter_channel_size, "
           << "int output_filter_size, "
           << "int output_pixels, "
           << "int output_pixels_magic, "
           << "int output_pixels_shift, "
           << "int* pad, "
           << "int* data_dilation, "
           << "int* data_dilation_magic, "
           << "int* data_dilation_shift, "
           << "int* filter_strides, "
           << "int* filter_dilation, "
           << "int* in_shape, "
           << "int* in_shape_str, "
           << "int* out_dim_str, "
           << "int* out_str_magic, "
           << "int* out_str_shift, "
           << "int* filter_dim_str, "
           << "int* filter_str_magic, "
           << "int* filter_str_shift"
           << ")\n";
    writer.block_begin();
    {
        writer << "Matrix* I = reinterpret_cast<Matrix*>(in);\n";
        writer << "Matrix* F = reinterpret_cast<Matrix*>(filter);\n";
        writer << "Matrix* O = reinterpret_cast<Matrix*>(out);\n";

        writer << "__shared__ int2 lookup_table[FILTER_SIZE];\n";
        writer << "__shared__ int lookup_size;\n";
        writer << "__shared__ Matrix a_tile[NUM_ROWS][SM_TILE_SIZE];\n";
        writer << "__shared__ Matrix b_tile[NUM_ROWS][SM_TILE_SIZE];\n";
        writer << "int lookup_size_local = 0;\n";

        writer << "int n_batch = division_by_invariant_multiplication(blockIdx.x, "
                  "output_pixels_magic, output_pixels_shift);\n";
        writer << "int output_pixel_idx = blockIdx.x - n_batch*output_pixels;\n";

        // multiply by the number of threads per sm tile to get the offset into the
        // image and filter dimensions (stride 1)
        writer << "int n_offset = n_batch * blockDim.x;\n";
        writer << "int k_offset = blockIdx.y * blockDim.x;\n";

        // compute coordinate transform to output tensor axes
        // up to the last dimension but not including it
        // : out_dim_str { d2*d3*...*dn, d3*...*dn, ..., dn, 1}
        // : for 2d {Q, 1}
        coordinate_transform_to_multi_d(writer,
                                        "out_dim_str",
                                        "out_str_magic",
                                        "out_str_shift",
                                        "output_pixel_idx",
                                        "out_d",
                                        rank);

        // offset tensors by image and filter indices
        // each thread is responsible for it's own image and filter

        // n and k offsets are required because only REG_TILE_SIZE*SM_TILE_SIZE
        // images/filters are processed per block
        writer << "I = &(I[n_offset + threadIdx.x]);\n";
        writer << "F = &(F[k_offset + threadIdx.x]);\n";

        // if N is a multiple of reg_tile_size * sm_tile_size then no check is needed
        bool need_image_bounds_check = N % (reg_tile_size * sm_tile_size) != 0;
        if (need_image_bounds_check)
        {
            writer << "int image_load_in_bounds = (n_offset + threadIdx.x);\n";
            if (reg_tile_size == 4)
            {
                writer << "image_load_in_bounds <<= 2;\n";
            }
            writer << "image_load_in_bounds = (image_load_in_bounds < N);\n";
        }

        // if K is a multiple of reg_tile_size * sm_tile_size then no check is needed
        bool need_filter_bounds_check = K % (reg_tile_size * sm_tile_size) != 0;
        if (need_filter_bounds_check)
        {
            writer << "int filter_load_in_bounds = (k_offset + threadIdx.x);\n";
            if (reg_tile_size == 4)
            {
                writer << "filter_load_in_bounds <<= 2;\n";
            }
            writer << "filter_load_in_bounds = (filter_load_in_bounds < K);\n";
        }

        writer << "int tid = threadIdx.x + threadIdx.y * blockDim.x;\n";
        // build lookup table for loading elements from data and filter tensors
        writer << "if (tid < 32)\n";
        writer.block_begin();
        {
            writer << "int filter_pixel = tid;\n";

            for (int i = 0; i < rank; i++)
            {
                writer << "int input_base_d" << i << " = out_d" << i << " * filter_strides[" << i
                       << "] - pad[" << i << "];\n";
            }

            // a mask marking all threads that have tid less than the current thread
            writer << "uint32_t mask = (1 << tid) - 1;\n";
            // loop over filter coordinates
            writer << "while (filter_pixel < FILTER_SIZE)\n";
            writer.block_begin();
            {
                // transform to filter coordinates
                // : filter_dim_str is {S, 1} for 2D
                coordinate_transform_to_multi_d(writer,
                                                "filter_dim_str",
                                                "filter_str_magic",
                                                "filter_str_shift",
                                                "filter_pixel",
                                                "filter_d",
                                                rank);
                // transform from filter coordinate to input coordinates
                // and check that each coordinate maps to an input element in the undilated space
                writer << "int off_dilation_stride = 0;\n";
                writer << "int undilated_coordinate = 0;\n";
                for (int i = 0; i < rank; i++)
                {
                    writer << "int input_d" << i << " = input_base_d" << i << " + filter_d" << i
                           << " * filter_dilation[" << i << "];\n";
                    // determine coordinate in undilated input space
                    writer << "undilated_coordinate = division_by_invariant_multiplication(input_d"
                           << i << ", data_dilation_magic[" << i << "], data_dilation_shift[" << i
                           << "]);\n";
                    // if division remainder is 0, then dilated coordinate is on an input element
                    writer << "off_dilation_stride += (input_d" << i
                           << " - undilated_coordinate * data_dilation[" << i << "]);\n";
                    // reassign dilated coordinate to undilated input coordinate
                    writer << "input_d" << i << " = undilated_coordinate;\n";
                }

                // check if the index is in bounds of the input tensor
                writer << "bool in_bounds = (off_dilation_stride == 0) && (";
                for (int i = 0; i < rank; i++)
                {
                    if (i != 0)
                    {
                        writer << "&& ";
                    }

                    // in_shape contains the full shape of the input_tensor
                    // for 2D this is: (C, H, W, N) but rank = 2 and so only [H, W] are used
                    // condition (input_d0 >=0 && input_d0 < H && input_d1 >= 0 && input_d1 < W)
                    writer << "input_d" << i << ">= 0 && input_d" << i << " < in_shape[" << i + 1
                           << "] ";
                }
                writer << ");\n";

                // check which threads are within bounds of the input tensor
                writer << "uint32_t threads = __ballot(in_bounds);\n";
                writer << "if (in_bounds)\n";
                writer.block_begin();
                {
                    writer << "int2 entry;\n";
                    // inner product of coordinates and strides up to the last dimension
                    // for 2D (CHWN) this is: (HWN, WN, N, 1)
                    // entry.x = input_d0 * WN + input_d1*N

                    writer << "entry.x = (";
                    for (int i = 0; i < rank; i++)
                    {
                        if (i != 0)
                        {
                            writer << "+ ";
                        }
                        // skips the first and last stride which correspond
                        // to the channel and batch coordinate, respectively
                        writer << "input_d" << i << " * in_shape_str[" << i + 1 << "] ";
                    }
                    writer << ")";
                    // if using register tiling, down shift
                    // as each thread will compute outer
                    // product with register tiles
                    if (reg_tile_size == 4)
                    {
                        writer << " >> 2";
                    }
                    writer << ";\n";

                    // multiply by K filters per filter_pixel
                    writer << "entry.y = (filter_pixel * K)";
                    if (reg_tile_size == 4)
                    {
                        writer << " >> 2";
                    }
                    writer << ";\n";

                    // count the number of active threads with index less than
                    // current tid use this as an offset into the lookup table
                    writer << "int index = lookup_size_local + __popc(threads & mask);\n";
                    // save coordinates to shared lookup table for later loading
                    writer << "lookup_table[index] = entry;\n";
                }
                writer.block_end();
                writer << "lookup_size_local += __popc(threads);\n";
                writer << "filter_pixel += 32;\n";
            }
            writer.block_end();
        }
        writer.block_end();

        // push lookup table size to shared memory so that it is accessible by other threads
        writer << "if (tid == 0)\n";
        writer.block_begin();
        {
            writer << "lookup_size = lookup_size_local;\n";
        }
        writer.block_end();
        writer << "__syncthreads();\n";
        // pull lookup table size from shared memory
        writer << "lookup_size_local = lookup_size;\n";
        // declare and zero initialize gemm accumulator
        writer << "Matrix result[" << reg_tile_size << "] = {0};\n";
        // if the lookup table is empty no multiplication is needed,
        // skip and write out zero result else, do the gemm
        writer << "if (lookup_size_local > 0)\n";
        writer.block_begin();
        {
            // calculate total size of filter including each channel
            writer << "int total_filter_size = lookup_size_local * C;\n";
            // precompute reciprocal for faster division
            writer << "float reciprocal = 1.0f / static_cast<float>(lookup_size_local);\n";
            // loop from the back of the filter (highest index) to the front
            // in order to handle filter pixel edge conditionals first (outside of gemm loop)
            writer << "int total_filter_idx = total_filter_size % NUM_ROWS;\n";
            // want total_filter_idx always >=0 in order to mask threads with t.y > total_filter_idx
            writer << "total_filter_idx = (total_filter_idx == 0) ? 8 : total_filter_idx;\n";

            // first iteration from back of filter
            writer << "int c;\n";
            writer << "int filter_idx;\n";
            writer << "idiv_fast(total_filter_size - threadIdx.y - 1, lookup_size_local, "
                      "reciprocal, c, filter_idx);\n";
            // retrieve the offsets for the data and filter for these filter pixels
            // only threads that are less than the total_filter_idx are valid, the rest are oob
            writer << "int2 entry = ((threadIdx.y & 7) >= total_filter_idx) "
                   << "? make_int2(0, 0)\n"
                   << ": lookup_table[filter_idx];\n";

            // helper to emit call to cuda make_float function
            auto make_float_i = [](int n) {
                std::stringstream ss;
                ss << "make_float" << n << "(";
                for (int i = 0; i < n; i++)
                {
                    if (i != 0)
                    {
                        ss << ", ";
                    }
                    ss << "0";
                }
                ss << ")";
                return ss.str();
            };

            // use the y index of threads to load data into the tile rows
            // threadIdx.x is used for iterating over the fastest moving dimensions
            // of the data and filter tensors (N and K respectively)

            // --- image load ---
            writer << "a_tile[threadIdx.y][threadIdx.x].f" << reg_tile_size << " =\n";
            if (need_image_bounds_check)
            {
                // check if image index is in bounds
                writer << "(!image_load_in_bounds)\n";
                writer << "? " << make_float_i(reg_tile_size) << "\n";
                writer << ": ";
            }
            // if filter pixel is out of range,
            // set all elements in the relevant sm tile row to 0
            writer << "((threadIdx.y & 7) >= total_filter_idx)\n";
            writer << "? " << make_float_i(reg_tile_size) << "\n";
            // else load the image data corresponding to this filter pixel
            // according to the entry.x offset previously determined
            writer << ": I[(c * input_channel_size) + entry.x].f" << reg_tile_size << ";\n";

            // --- filter load ---
            writer << "b_tile[threadIdx.y][threadIdx.x].f" << reg_tile_size << " =\n";
            if (need_filter_bounds_check)
            {
                // check if filter index is in bounds
                writer << "(!filter_load_in_bounds)\n";
                writer << "? " << make_float_i(reg_tile_size) << "\n";
                writer << ": ";
            }
            // if filter pixel is out of range,
            // set all elements in the relevant sm tile row to 0
            writer << "((threadIdx.y & 7) >= total_filter_idx)\n";
            writer << "? " << make_float_i(reg_tile_size) << "\n";
            // else load the filter weights corresponding to this filter pixel
            // according to the entry.y offset previously determined
            writer << ": F[(c * filter_channel_size) + entry.y].f" << reg_tile_size << ";\n";

            // iterate over filter from back to front
            writer << "for (total_filter_idx = total_filter_size - total_filter_idx; "
                      "total_filter_idx > 0; total_filter_idx -= NUM_ROWS)\n";
            writer.block_begin();
            {
                // finish loads
                writer << "__syncthreads();\n";
                writer << "#pragma unroll\n";
                writer << "for (int i = 0; i < NUM_ROWS; i++)\n";
                writer.block_begin();
                {
                    writer << "Matrix row;\n";
                    writer << "Matrix col;\n";
                    writer << "row.f" << reg_tile_size << " = a_tile[i][threadIdx.x].f"
                           << reg_tile_size << ";\n";
                    writer << "col.f" << reg_tile_size << " = b_tile[i][threadIdx.y].f"
                           << reg_tile_size << ";\n";

                    // accumulate the product
                    writer << "#pragma unroll\n";
                    writer << "for (int y = 0; y < " << reg_tile_size << "; y++)\n";
                    writer.block_begin();
                    {
                        writer << "#pragma unroll\n";
                        writer << "for (int x = 0; x < " << reg_tile_size << "; x++)\n";
                        writer.block_begin();
                        {
                            writer << "result[y].f[x] += (row.f[x] * col.f[y]);\n";
                        }
                        writer.block_end();
                    }
                    writer.block_end();
                }
                writer.block_end();
                writer << "__syncthreads();\n";

                // load new data and weights
                writer << "idiv_fast(total_filter_idx - threadIdx.y - 1, lookup_size_local, "
                          "reciprocal, c, filter_idx);\n";
                writer << "entry = lookup_table[filter_idx];\n";

                // --- image load ---
                writer << "a_tile[threadIdx.y][threadIdx.x].f" << reg_tile_size << " =\n";
                if (need_image_bounds_check)
                {
                    // check if image index is in bounds
                    writer << "(!image_load_in_bounds)\n";
                    writer << "? " << make_float_i(reg_tile_size) << "\n";
                    writer << ": ";
                }
                writer << "I[(c * input_channel_size) + entry.x].f" << reg_tile_size << ";\n";

                // --- filter load ---
                writer << "b_tile[threadIdx.y][threadIdx.x].f" << reg_tile_size << " =\n";
                if (need_filter_bounds_check)
                {
                    // check if filter index is in bounds
                    writer << "(!filter_load_in_bounds)\n";
                    writer << "? " << make_float_i(reg_tile_size) << "\n";
                    writer << ": ";
                }
                writer << "F[(c * filter_channel_size) + entry.y].f" << reg_tile_size << ";\n";
            }
            writer.block_end();
            writer << "__syncthreads();\n";

            // last iteration
            writer << "#pragma unroll\n";
            writer << "for (int i = 0; i < NUM_ROWS; i++)\n";
            writer.block_begin();
            {
                writer << "Matrix row;\n";
                writer << "Matrix col;\n";
                writer << "row.f" << reg_tile_size << " = a_tile[i][threadIdx.x].f" << reg_tile_size
                       << ";\n";
                writer << "col.f" << reg_tile_size << " = b_tile[i][threadIdx.y].f" << reg_tile_size
                       << ";\n";
                // accumulate the product
                writer << "#pragma unroll\n";
                writer << "for (int y = 0; y < " << reg_tile_size << "; y++)\n";
                writer.block_begin();
                {
                    writer << "#pragma unroll\n";
                    writer << "for (int x = 0; x < " << reg_tile_size << "; x++)\n";
                    writer.block_begin();
                    {
                        writer << "result[y].f[x] += (row.f[x] * col.f[y]);\n";
                    }
                    writer.block_end();
                }
                writer.block_end();
            }
            writer.block_end();
        } // end if (lookup_size_local > 0)
        writer.block_end();

        // store result block to global memory
        writer << "int n = n_offset + threadIdx.x;\n";
        std::string k_definition = "int k = (k_offset + threadIdx.y)";
        std::string output_pixel = "output_pixel_idx = (output_pixel_idx * N)";
        if (reg_tile_size == 4)
        {
            output_pixel += " >> 2";
            k_definition += " << 2";
        }
        writer << output_pixel << ";\n";
        writer << k_definition << ";\n";
        writer << "if (k < K && n < N)\n";
        writer.block_begin();
        {
            writer << "#pragma unroll\n";
            writer << "for (int x = 0; x < " << reg_tile_size << "; x++)\n";
            writer.block_begin();
            {
                writer << "if (k < K)\n";
                writer.block_begin();
                {
                    writer << "int idx = (k * output_filter_size) + output_pixel_idx + n;\n";
                    writer << "O[idx].f" << reg_tile_size << " = result[x].f" << reg_tile_size
                           << ";\n";
                }
                writer.block_end();
                writer << "k++;\n";
            }
            writer.block_end();
        }
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::CudaKernelBuilder::coordinate_transform_to_multi_d(codegen::CodeWriter& writer,
                                                                      std::string i_strides,
                                                                      std::string i_stride_magic,
                                                                      std::string i_stride_shift,
                                                                      std::string i_coord_product,
                                                                      std::string o_coordinates,
                                                                      size_t rank)
{
    // Translation from flat index to dense tensor coordinates:
    // Given tensor shape [d0 d1 ... dN] with strides [d1*...*dN, d2*...*dN, ... 1],
    // calculate coordinates as:
    //
    //  product = tid
    //  d0 = product/stride[0]
    //  product = product % stride[0]
    //  d1 = product/stride[1]
    //  ...
    writer << "int coordinate_product = " << i_coord_product << ";\n";
    for (size_t i = 0; i < rank; i++)
    {
        if (i != 0)
        {
            writer << "coordinate_product -= (" << o_coordinates << i - 1 << " * " << i_strides
                   << "[" << i - 1 << "]);\n";
        }
        writer << "int " << o_coordinates << i << " = division_by_invariant_multiplication("
               << "coordinate_product, " << i_stride_magic << "[" << i << "], " << i_stride_shift
               << "[" << i << "]);\n";
    }
}
std::string runtime::gpu::CudaKernelBuilder::collective_coordinate_transform_helper(
    codegen::CodeWriter& writer,
    std::string i_thread_index,
    std::string i_strides,
    std::string i_stride_magic,
    std::string i_stride_shift,
    std::string i_reduced_strides,
    std::string o_coordinates,
    size_t rank)
{
    coordinate_transform_to_multi_d(
        writer, i_strides, i_stride_magic, i_stride_shift, i_thread_index, o_coordinates, rank);

    // index into reduced tensor from coordinates of non-reduced tensor
    std::string reduced_idx = "reduced_idx";
    writer << "int " << reduced_idx << " = 0;\n";
    for (size_t i = 0; i < rank; i++)
    {
        writer << "reduced_idx += " << o_coordinates << i << " * " << i_reduced_strides << "[" << i
               << "];\n";
    }

    return reduced_idx;
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
