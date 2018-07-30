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

            void emit_onehot(const std::string& name,
                             std::array<std::string, 2> data_types,
                             GPURuntimeContext* ctx,
                             CUdeviceptr in,
                             CUdeviceptr out,
                             size_t repeat_size,
                             size_t repeat_times,
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
        }
    }
}
