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

void runtime::gpu::emit_broadcast(std::string name,
                                  CUdeviceptr in,
                                  CUdeviceptr out,
                                  std::array<std::string, 2> data_types,
                                  size_t repeat_size,
                                  size_t repeat_times,
                                  size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    // Create an instance of nvrtcProgram with the code string.
    if (CudaFunctionPool::instance().get(name_signature) == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_broadcast_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        CudaFunctionPool::instance().set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &repeat_size, &repeat_times, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*CudaFunctionPool::instance().get(name_signature).get(),
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

void runtime::gpu::emit_onehot(std::string name,
                               CUdeviceptr in,
                               CUdeviceptr out,
                               std::array<std::string, 2> data_types,
                               size_t repeat_size,
                               size_t repeat_times,
                               size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    // Create an instance of nvrtcProgram with the code string.
    if (CudaFunctionPool::instance().get(name_signature) == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_onehot_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        CudaFunctionPool::instance().set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &repeat_size, &repeat_times, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*CudaFunctionPool::instance().get(name_signature).get(),
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

void runtime::gpu::emit_reshape(std::string name,
                    CUdeviceptr in,
                    CUdeviceptr out,
                    std::array<std::string, 2> data_types,
                    CUdeviceptr input_stride,
                    CUdeviceptr output_stride,
                    size_t rank,
                    size_t count)
{
    std::string name_signature = name + "_" + data_types[0] + "_" + data_types[1];
    std::replace(name_signature.begin(), name_signature.end(), ' ', '_');
    // Create an instance of nvrtcProgram with the code string.
    if (CudaFunctionPool::instance().get(name_signature) == nullptr)
    {
        codegen::CodeWriter writer;
        CudaKernelBuilder::add_pod_typedefs(writer);
        CudaKernelBuilder::get_reshape_op(writer, name_signature, data_types);
        std::string kernel = writer.get_code();
        CudaFunctionPool::instance().set(name_signature, kernel);
    }

    void* args_list[] = {&in, &out, &input_stride, &output_stride, &rank, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*CudaFunctionPool::instance().get(name_signature).get(),
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
