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

#include <array>
#include <string>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <typename T>
            struct CudaOpMap;

            void emit_broadcast(const std::string& name,
                                std::array<std::string, 2> data_types,
                                GPURuntimeContext* ctx,
                                CUdeviceptr in,
                                CUdeviceptr out,
                                size_t repeat_size,
                                size_t repeat_times,
                                size_t count);

            void emit_onehot(const std::string& name,
                             std::array<std::string, 2> data_types,
                             GPURuntimeContext* ctx,
                             CUdeviceptr in,
                             CUdeviceptr out,
                             size_t repeat_size,
                             size_t repeat_times,
                             size_t count);

            void emit_reshape(const std::string& name,
                              const std::array<std::string, 2>& data_types,
                              GPURuntimeContext* ctx,
                              CUdeviceptr in,
                              CUdeviceptr out,
                              CUdeviceptr input_strides,
                              CUdeviceptr trans_strides,
                              size_t rank,
                              size_t count);

            void emit_slice(const std::string& name,
                            CUdeviceptr in,
                            CUdeviceptr out,
                            const std::array<std::string, 2>& data_types,
                            GPURuntimeContext* ctx,
                            CUdeviceptr input_strides,
                            CUdeviceptr lower_bounds,
                            CUdeviceptr slice_strides,
                            CUdeviceptr output_strides,
                            size_t rank,
                            size_t count);

            void emit_reverse(const std::string& name,
                              CUdeviceptr in,
                              CUdeviceptr out,
                              const std::array<std::string, 2>& data_types,
                              GPURuntimeContext* ctx,
                              CUdeviceptr input_shape,
                              CUdeviceptr reverse_axes,
                              size_t rank,
                              size_t count);

            template <typename T, typename... Inputs>
            void emit_elementwise_op(const std::string& name,
                                     const std::vector<std::string>& data_types,
                                     GPURuntimeContext* ctx,
                                     size_t count,
                                     CUdeviceptr out,
                                     Inputs&&... inputs)
            {
                std::string type_signature = "_" + join(data_types, "_");
                std::replace(type_signature.begin(), type_signature.end(), ' ', '_');
                auto compiled_kernel = ctx->compiled_kernel_pool->get(name + type_signature);
                if (compiled_kernel == nullptr)
                {
                    codegen::CodeWriter writer;
                    CudaKernelBuilder::add_pod_typedefs(writer);

                    std::string op_name = CudaOpMap<T>::op;
                    if (CudaOpMap<T>::math_kernel)
                    {
                        op_name += type_signature;
                        CudaKernelBuilder::get_device_helper(writer,
                                                             op_name,
                                                             CudaOpMap<T>::math_kernel,
                                                             data_types,
                                                             sizeof...(inputs));
                    }

                    CudaKernelBuilder::get_elementwise_op(
                        writer, name + type_signature, op_name, data_types, sizeof...(inputs));

                    std::string kernel = writer.get_code();
                    compiled_kernel = ctx->compiled_kernel_pool->set(name + type_signature, kernel);
                }

                void* args_list[] = {&inputs..., &out, &count};
                CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
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

            template <typename... Inputs>
            void emit_concat_op(const std::string& name,
                                const std::vector<std::string>& data_types,
                                GPURuntimeContext* ctx,
                                size_t count,
                                size_t block_size,
                                CUdeviceptr block_strides,
                                CUdeviceptr out,
                                Inputs&&... inputs)
            {
                std::string type_signature = "_" + join(data_types, "_");
                std::replace(type_signature.begin(), type_signature.end(), ' ', '_');
                auto compiled_kernel = ctx->compiled_kernel_pool->get(name + type_signature);
                if (compiled_kernel == nullptr)
                {
                    codegen::CodeWriter writer;
                    CudaKernelBuilder::add_pod_typedefs(writer);

                    CudaKernelBuilder::get_concat_op(
                        writer, name + type_signature, data_types, sizeof...(inputs));

                    std::string kernel = writer.get_code();
                    compiled_kernel = ctx->compiled_kernel_pool->set(name + type_signature, kernel);
                }

                void* args_list[] = {&inputs..., &out, &block_strides, &block_size, &count};
                CUDA_SAFE_CALL(cuLaunchKernel(*compiled_kernel.get(),
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
