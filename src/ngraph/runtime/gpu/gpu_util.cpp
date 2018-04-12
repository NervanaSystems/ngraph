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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

void runtime::gpu::print_gpu_f32_tensor(void* p, size_t element_count, size_t element_size)
{
    std::vector<float> local(element_count);
    size_t size_in_bytes = element_size * element_count;
    cudaMemcpy(local.data(), p, size_in_bytes, cudaMemcpyDeviceToHost);
    std::cout << "{" << join(local) << "}" << std::endl;
}

void runtime::gpu::check_cuda_errors(CUresult err)
{
    assert(err == CUDA_SUCCESS);
}

void* runtime::gpu::create_gpu_buffer(size_t buffer_size)
{
    void* allocated_buffer_pool;
    cudaMalloc(static_cast<void**>(&allocated_buffer_pool), buffer_size);
    return allocated_buffer_pool;
}

void runtime::gpu::free_gpu_buffer(void* buffer)
{
    if (buffer)
    {
        cudaFree(buffer);
    }
}

void runtime::gpu::cuda_memcpyDtD(void* dst, void* src, size_t buffer_size)
{
    cudaMemcpy(dst, src, buffer_size, cudaMemcpyDeviceToDevice);
}

void runtime::gpu::cuda_memcpyHtD(void* dst, void* src, size_t buffer_size)
{
    cudaMemcpy(dst, src, buffer_size, cudaMemcpyHostToDevice);
}

void runtime::gpu::cuda_memset(void* dst, int value, size_t buffer_size)
{
    cudaMemset(dst, value, buffer_size);
}
