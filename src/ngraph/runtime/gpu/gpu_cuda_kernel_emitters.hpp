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

#pragma once

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <typename T>
            struct CudaOpMap;

            void emit_broadcast(
                void* in, void* out, size_t repeat_size, size_t repeat_times, size_t count);

            template <typename T, typename... Inputs>
            void emit_elementwise_op(std::string name,
                                     size_t count,
                                     CUdeviceptr out,
                                     Inputs&&... inputs)
            {
                // Create an instance of nvrtcProgram with the code string.
                if (CudaFunctionPool::instance().get(name) == nullptr)
                {
                    codegen::CodeWriter writer;
                    CudaKernelBuilder::get_elementwise_op(writer,
                        name, CudaOpMap<T>::type, CudaOpMap<T>::op, sizeof...(inputs));

                    std::string kernel = writer.get_code();
                    CudaFunctionPool::instance().set(name, kernel);
                }

                //convert runtime ptr to driver api ptr
                void* args_list[] = {&inputs..., &out, &count};
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
