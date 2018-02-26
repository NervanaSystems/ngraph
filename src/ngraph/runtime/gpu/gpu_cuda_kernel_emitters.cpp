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
            namespace cuda
            {
                namespace kernel
                {
                    void emit_abs(void* in, void* out, size_t count)
                    {
                        std::string name = "abs";
                        // Create an instance of nvrtcProgram with the code string.
                        if (Cuda_function_pool::Instance().Get(name) == nullptr)
                        {
                            const char* opts[] = {"--gpu-architecture=compute_35",
                                                  "--relocatable-device-code=true"};
                            std::string kernel;
                            Cuda_kernel_builder::Get_1_element_op(name, "float", "fabsf", kernel);
                            Cuda_function_pool::Instance().Set(
                                name, CudaFunctionBuilder::Get("cuda_" + name, kernel, 2, opts));
                        }

                        //convert runtime ptr to driver api ptr
                        CUdeviceptr dPtrIn, dPtrOut;
                        dPtrIn = (CUdeviceptr)in;
                        dPtrOut = (CUdeviceptr)out;

                        void* argsList[] = {&dPtrIn, &dPtrOut, &count};
                        CUDA_SAFE_CALL(
                            cuLaunchKernel(*Cuda_function_pool::Instance().Get(name).get(),
                                           count,
                                           1,
                                           1, // grid dim
                                           1,
                                           1,
                                           1, // block dim
                                           0,
                                           NULL, // shared mem and stream
                                           argsList,
                                           0));             // arguments
                        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
                    }
                }
            }
        }
    }
}
