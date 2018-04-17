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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn_v7.h>
#include <nvrtc.h>

//why use "do...while.."
//https://stackoverflow.com/questions/154136/why-use-apparently-meaningless-do-while-and-if-else-statements-in-macros
#define NVRTC_SAFE_CALL(x)                                                                         \
    do                                                                                             \
    {                                                                                              \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS)                                                               \
        {                                                                                          \
            throw std::runtime_error("\nerror: " #x " failed with error " +                        \
                                     std::string(nvrtcGetErrorString(result)));                    \
        }                                                                                          \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS)                                                                \
        {                                                                                          \
            const char* msg;                                                                       \
            cuGetErrorName(result, &msg);                                                          \
            throw std::runtime_error("\nerror: " #x " failed with error " + std::string(msg));     \
        }                                                                                          \
    } while (0)

#define CUDNN_SAFE_CALL(func)                                                                      \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            auto msg = cudnnGetErrorString(e);                                                     \
            throw std::runtime_error("\ncuDNN error: " + std::string(msg));                        \
        }                                                                                          \
    }

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void print_gpu_f32_tensor(void* p, size_t element_count, size_t element_size);
            void check_cuda_errors(CUresult err);
            void* create_gpu_buffer(size_t buffer_size);
            void free_gpu_buffer(void* buffer);
            void cuda_memcpyDtD(void* dst, void* src, size_t buffer_size);
            void cuda_memcpyHtD(void* dst, void* src, size_t buffer_size);
            void cuda_memset(void* dst, int value, size_t buffer_size);
        }
    }
}
