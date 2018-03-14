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

void runtime::gpu::emit_broadcast(
    void* in, void* out, size_t repeat_size, size_t repeat_times, size_t count)
{
    std::string name = "broadcast";
    // Create an instance of nvrtcProgram with the code string.
    if (CudaFunctionPool::instance().get(name) == nullptr)
    {
        std::string kernel;
        std::string data_type("float");

        kernel = R"(
extern "C" __global__
void cuda_)" + name +
                 "(" + data_type + "* in, " + data_type + "* out, size_t m, size_t k, size_t n)\n" +
                 R"(
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        size_t idx = tid / (m * k) * m + tid % m;
        out[tid] = in[idx];
    }
})";
        CudaFunctionPool::instance().set(name, kernel);
    }

    //convert runtime ptr to driver api ptr
    CUdeviceptr d_ptr_in, d_ptr_out;
    d_ptr_in = CUdeviceptr(in);
    d_ptr_out = CUdeviceptr(out);

    void* args_list[] = {&d_ptr_in, &d_ptr_out, &repeat_size, &repeat_times, &count};
    CUDA_SAFE_CALL(cuLaunchKernel(*CudaFunctionPool::instance().get(name).get(),
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
