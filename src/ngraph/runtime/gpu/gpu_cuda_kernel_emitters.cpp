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

#include "ngraph/runtime/gpu/gpu_cuda_function_builder.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void emit_abs(void* in, void* out, size_t count)
            {
                std::string name = "abs";
                // Create an instance of nvrtcProgram with the code string.
                if (CudaFunctionPool::instance().get(name) == nullptr)
                {
                    const char* opts[] = {"--gpu-architecture=compute_35",
                                          "--relocatable-device-code=true"};
                    std::string kernel;
                    CudaKernelBuilder::get_1_element_op(name, "float", "fabsf", kernel);
                    CudaFunctionPool::instance().set(
                        name, CudaFunctionBuilder::get("cuda_" + name, kernel, 2, opts));
                }

                //convert runtime ptr to driver api ptr
                CUdeviceptr d_ptr_in, d_ptr_out;
                d_ptr_in = (CUdeviceptr)in;
                d_ptr_out = (CUdeviceptr)out;

                void* args_list[] = {&d_ptr_in, &d_ptr_out, &count};
                CUDA_SAFE_CALL(cuLaunchKernel(*CudaFunctionPool::instance().get(name).get(),
                                              count,
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
        }
    }
}
