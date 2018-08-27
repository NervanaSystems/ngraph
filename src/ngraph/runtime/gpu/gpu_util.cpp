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
#include <stddef.h>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

void runtime::gpu::print_gpu_f32_tensor(const void* p, size_t element_count, size_t element_size)
{
    std::vector<float> local(element_count);
    size_t size_in_bytes = element_size * element_count;
    CUDA_RT_SAFE_CALL(cudaMemcpy(local.data(), p, size_in_bytes, cudaMemcpyDeviceToHost));
    std::cout << "{" << join(local) << "}" << std::endl;
}

void runtime::gpu::check_cuda_errors(CUresult err)
{
    assert(err == CUDA_SUCCESS);
}

void* runtime::gpu::create_gpu_buffer(size_t buffer_size, const void* data)
{
    void* allocated_buffer_pool;
    CUDA_RT_SAFE_CALL(cudaMalloc(static_cast<void**>(&allocated_buffer_pool), buffer_size));
    if (data)
    {
        runtime::gpu::cuda_memcpyHtD(allocated_buffer_pool, data, buffer_size);
    }
    return allocated_buffer_pool;
}

void runtime::gpu::free_gpu_buffer(void* buffer)
{
    if (buffer)
    {
        CUDA_RT_SAFE_CALL(cudaFree(buffer));
    }
}

void runtime::gpu::cuda_memcpyDtD(void* dst, const void* src, size_t buffer_size)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(dst, src, buffer_size, cudaMemcpyDeviceToDevice));
}

void runtime::gpu::cuda_memcpyHtD(void* dst, const void* src, size_t buffer_size)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(dst, src, buffer_size, cudaMemcpyHostToDevice));
}

void runtime::gpu::cuda_memcpyDtH(void* dst, const void* src, size_t buffer_size)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(dst, src, buffer_size, cudaMemcpyDeviceToHost));
}

void runtime::gpu::cuda_memset(void* dst, int value, size_t buffer_size)
{
    CUDA_RT_SAFE_CALL(cudaMemset(dst, value, buffer_size));
}

namespace
{
    // Unsigned integer exponentiation by squaring adapted
    // from https://stackoverflow.com/a/101613/882253
    uint64_t powU64(uint64_t base, uint64_t exp)
    {
        uint64_t result = 1;
        do
        {
            if (exp & 1)
            {
                result *= base;
            }
            exp >>= 1;
            if (!exp)
            {
                break;
            }
            base *= base;
        } while (true);

        return result;
    }

    // Most significant bit search via de bruijn multiplication
    // Adopted from https://stackoverflow.com/a/31718095/882253
    // Additional ref: http://supertech.csail.mit.edu/papers/debruijn.pdf
    uint32_t msbDeBruijnU32(uint32_t v)
    {
        static const int multiply_de_Bruijn_bit_position[32] = {
            0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
            8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31};

        v |= v >> 1; // first round down to one less than a power of 2
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;

        return multiply_de_Bruijn_bit_position[static_cast<uint32_t>(v * 0x07C4ACDDU) >> 27];
    }

    // perform msb on upper 32 bits if the first 32 bits are filled
    // otherwise do normal de bruijn mutliplication on the 32 bit word
    int msbU64(uint64_t val)
    {
        if (val > 0x00000000FFFFFFFFul)
        {
            return 32 + msbDeBruijnU32(static_cast<uint32_t>(val >> 32));
        }
        // Number is no more than 32 bits,
        // so calculate number of bits in the bottom half.
        return msbDeBruijnU32(static_cast<uint32_t>(val & 0xFFFFFFFF));
    }

    // Magic numbers and shift amounts for integer division
    // Suitable for when nmax*magic fits in 32 bits
    // Translated from http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
    std::pair<uint64_t, uint64_t> magicU32(uint64_t nmax, uint64_t d)
    {
        uint64_t nc = ((nmax + 1) / d) * d - 1;
        uint64_t nbits = msbU64(nmax) + 1;

        for (uint64_t p = 0; p < 2 * nbits + 1; p++)
        {
            uint64_t pow2 = powU64(2, p);
            if (pow2 > nc * (d - 1 - (pow2 - 1) % d))
            {
                uint64_t m = (pow2 + d - 1 - (pow2 - 1) % d) / d;
                return std::pair<uint64_t, uint64_t>{m, p};
            }
        }
        throw std::runtime_error("Magic for unsigned integer division could not be found.");
    }

    // Magic numbers and shift amounts for integer division
    // Suitable for when nmax*magic fits in 64 bits and the shift
    // lops off the lower 32 bits
    std::pair<uint64_t, uint64_t> magicU64(uint64_t d)
    {
        // 3 is a special case that only ends up in the high bits
        // if the nmax is 0xffffffff
        // we can't use 0xffffffff for all cases as some return a 33 bit
        // magic number
        uint64_t nmax = (d == 3) ? 0xffffffff : 0x7fffffff;
        uint64_t magic, shift;
        std::tie(magic, shift) = magicU32(nmax, d);
        if (magic != 1)
        {
            shift -= 32;
        }
        return std::pair<uint64_t, uint64_t>{magic, shift};
    }
}

std::pair<uint64_t, uint64_t> runtime::gpu::idiv_magic_u32(uint64_t max_numerator, uint64_t divisor)
{
    return magicU32(max_numerator, divisor);
}

std::pair<uint64_t, uint64_t> runtime::gpu::idiv_magic_u64(uint64_t divisor)
{
    return magicU64(divisor);
}

uint32_t runtime::gpu::idiv_ceil(int n, int d)
{
    // compiler fused modulo and division
    return n / d + (n % d > 0);
}

void runtime::gpu::StopWatch::start()
{
    if (m_active == false)
    {
        m_total_count++;
        m_active = true;
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEventRecord(start);
        starts.push_back(start);
    }
}

void runtime::gpu::StopWatch::stop()
{
    if (m_active == true)
    {
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop);
        stops.push_back(stop);
        m_active = false;
    }
}
size_t runtime::gpu::StopWatch::get_call_count()
{
    return m_total_count;
}

size_t runtime::gpu::StopWatch::get_total_seconds()
{
    return runtime::gpu::StopWatch::get_total_nanoseconds() / 1e9;
}

size_t runtime::gpu::StopWatch::get_total_milliseconds()
{
    return runtime::gpu::StopWatch::get_total_nanoseconds() / 1e6;
}

size_t runtime::gpu::StopWatch::get_total_microseconds()
{
    return runtime::gpu::StopWatch::get_total_nanoseconds() / 1e3;
}

size_t runtime::gpu::StopWatch::get_total_nanoseconds()
{
    // only need to sync the last stop.
    cudaEventSynchronize(stops.back());
    float total_time = 0;
    for (int i = 0; i < stops.size(); i++)
    {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, starts[i], stops[i]);
        total_time += milliseconds;
    }
    m_total_time_in_ns = static_cast<size_t>(total_time * 1000000.0f);
    return m_total_time_in_ns;
}
