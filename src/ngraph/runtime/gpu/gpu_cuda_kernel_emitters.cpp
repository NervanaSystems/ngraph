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
#include <map>

#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

void runtime::gpu::emit_broadcast(const std::string& name,
                                  std::array<std::string, 2> data_types,
                                  GPURuntimeContext* ctx,
                                  CUdeviceptr in,
                                  CUdeviceptr out,
                                  size_t repeat_size,
                                  size_t repeat_times,
                                  size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    // Create an instance of nvrtcProgram with the code string.
    auto compiled_kernel = ctx->nvrtc_cache->get(name_signature);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_broadcast_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        compiled_kernel = ctx->nvrtc_cache->set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &repeat_size, &repeat_times, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                  static_cast<unsigned int>(count),
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
}

void runtime::gpu::emit_onehot(const std::string& name,
                               std::array<std::string, 2> data_types,
                               GPURuntimeContext* ctx,
                               CUdeviceptr in,
                               CUdeviceptr out,
                               size_t repeat_size,
                               size_t repeat_times,
                               size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    // Create an instance of nvrtcProgram with the code string.
    auto compiled_kernel = ctx->nvrtc_cache->get(name_signature);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_onehot_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        compiled_kernel = ctx->nvrtc_cache->set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &repeat_size, &repeat_times, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                  static_cast<unsigned int>(count),
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
}

void runtime::gpu::emit_reshape(const std::string& name,
                                const std::array<std::string, 2>& data_types,
                                GPURuntimeContext* ctx,
                                CUdeviceptr in,
                                CUdeviceptr out,
                                CUdeviceptr input_strides,
                                CUdeviceptr trans_strides,
                                size_t rank,
                                size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    auto compiled_kernel = ctx->nvrtc_cache->get(name_signature);
    if (compiled_kernel == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_reshape_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        compiled_kernel = ctx->nvrtc_cache->set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &input_strides, &trans_strides, &rank, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
                                  static_cast<unsigned int>(count),
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
}

// void runtime::gpu::emit_1d_max_pool(GPURuntimeContext* ctx,
//                                     const std::string& name,
//                                     const std::string& kernel,
//                                     const std::array<std::string, 2>& data_types,
//                                     size_t count,
//                                     CUdeviceptr in,
//                                     CUdeviceptr out,
//                                     size_t width,
//                                     size_t stride,
//                                     size_t input_size,
//                                     size_t output_size)
// {
//     std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
//     std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
//     auto compiled_kernel = ctx->nvrtc_cache->get(name_signature);
//     if (compiled_kernel == nullptr)
//     {
//         compiled_kernel = ctx->nvrtc_cache->set(name_signature, kernel);
//     }

//     void* args_list[] = {&in, &out, &input_strides, &trans_strides, &rank, &count};
//     CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
//                                   static_cast<unsigned int>(count),
//                                   1,
//                                   1, // grid dim
//                                   1,
//                                   1,
//                                   1, // block dim
//                                   0,
//                                   NULL, // shared mem and stream
//                                   args_list,
//                                   0));  // arguments
//     CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
// }
