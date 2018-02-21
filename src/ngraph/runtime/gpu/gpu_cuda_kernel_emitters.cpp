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



#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_cude_kernel_builder.hpp"
#include "ngraph/runtime/gpu/gpu_cude_function_builder.hpp"
#include "ngraph/runtime/gpu/gpu_cude_function_pool.hpp"

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
                        if(Cuda_function_pool::Instance().get(name) == nullptr)
                        {
                            const char *opts[] = {"--gpu-architecture=compute_35",
                            "--relocatable-device-code=true"};
                            std::string kernel;
                            Cuda_kernel_builder::get_1_element_op(name, "float", "fabsf",kernel);
                            Cuda_function_pool::Instance().set(name, Cuda_function_builder(name, kernel, 2, opts));
                        }
                     
                        //convert runtime ptr to driver api ptr
                        CUdeviceptr dPtrIn, dPtrOut;
                        dPtrIn = (CUdeviceptr)in;
                        dPtrOut = (CUdeviceptr)out;

                        void *argsList[] = {&dPtrIn, &dPtrOut, &count};
                        CUDA_SAFE_CALL(
                                cuLaunchKernel(cudCuda_function_pool::Instance().get(name).get(), 
                                    count ,1, 1, // grid dim 
                                    1, 1, 1, // block dim 
                                    0, NULL, // shared mem and stream 
                                    argsList, 0)); // arguments 
                        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output. 
                    }

                    void emit_broadcast(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& broadcast_axes)
                    {
                    }

                    //
                    // For the reference kernel this is gpud on, see ngraph/runtime/kernel/concat.hpp.
                    //
                    void emit_concat(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::vector<std::string>& args,
                            const std::string& out,
                            const std::vector<Shape>& in_shapes,
                            const Shape& out_shape,
                            size_t concatenation_axis)
                    {
                    }

                    void emit_replace_slice(
                            codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& arg1, // replacement value
                            const std::string& out,
                            const Shape& arg1_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_slice(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_reshape(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisVector& arg0_axis_order)
                    {
                    }

                    void emit_sum(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& reduction_axes)
                    {
                    }

                }
            }
        }
    }
}
