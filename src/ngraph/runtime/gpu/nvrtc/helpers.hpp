//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <cuda.h>
#include <sstream>
#include <string>

#include "ngraph/except.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace nvrtc
            {
                std::string helpers();
                std::string define_zero(const std::string& dtype,
                                        const std::string& name = "zero_");
                std::string define_vzero(std::string dtype,
                                         const uint32_t& n,
                                         const std::string& name = "zero_");
                std::string define_coherent_load(const std::string& dtype,
                                                 const std::string& name = "load_");
                std::string define_coherent_vload(const std::string& dtype,
                                                  const uint32_t& n,
                                                  const std::string& name = "load_");
                std::string define_non_coherent_load(const std::string& dtype,
                                                     const std::string& name = "load_");
                std::string define_non_coherent_vload(const std::string& dtype,
                                                      const uint32_t& n,
                                                      const std::string& name = "load_");
            }
        }
    }
}

std::string ngraph::runtime::gpu::nvrtc::helpers()
{
    std::stringstream ss;
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
    ss << R"(
#define WARP_SIZE 32
#define __ballot_sync(mask, predicate) __ballot(predicate)
#define __shfl_down_sync(mask, val, delta, width) __shfl_down(val, delta, width)
#define __shfl_xor_sync(mask, val, laneMask, width) __shfl_xor(val, laneMask, width)
)";
#endif

    // add modern type definitions
    ss << "typedef signed char int8_t;\n";
    ss << "typedef signed short int16_t;\n";
    ss << "typedef signed int int32_t;\n";
    ss << "typedef signed long int int64_t;\n";
    ss << "typedef unsigned char uint8_t;\n";
    ss << "typedef unsigned short uint16_t;\n";
    ss << "typedef unsigned int uint32_t;\n";
    ss << "typedef unsigned long int uint64_t;\n";
    ss << "\n";

    // division_by_invariant_multiplication:
    // fast integer division via invariant multiplication and shifting
    // if value is a power of 2, magic will be 1 and only shifting
    // is required (predicate p below)
    // load: helper to load from constant memory for fast access
    ss << R"(
__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__device__ __forceinline__ void idiv_fast(int numerator, int denominator, float rcp,
                                          int& result, int& remainder)
{
    result = (int)((float)numerator * rcp);
    remainder = numerator - (result * denominator);
    result = (remainder >= denominator) ? (result + 1) : result;
    remainder = (remainder >= denominator) ? (remainder - denominator) : remainder;
}

__device__ __forceinline__ int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
__device__ __forceinline__ int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
__device__ __forceinline__ int msub16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, -%3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
)";
    return ss.str();
}

std::string ngraph::runtime::gpu::nvrtc::define_zero(const std::string& dtype,
                                                     const std::string& name)
{
    std::stringstream ss;
    ss << "__device__ __forceinline__ void " << name << "(" << dtype << "& a) { a = 0; }\n";
    return ss.str();
}

std::string ngraph::runtime::gpu::nvrtc::define_vzero(std::string dtype,
                                                      const uint32_t& n,
                                                      const std::string& name)
{
    std::stringstream ss;
    if (n == 1 || n == 2 || n == 4)
    {
        static std::vector<std::string> assignment = {"a.x = ", "a.y = ", "a.z = "};
        dtype = dtype + std::to_string(n);
        ss << "__device__ __forceinline__ void " << name << "(" << dtype << "& a) { ";
        for (auto i = 0u; i <= (n >> 1); i++)
        {
            ss << assignment[i];
        }
        ss << "0; }\n";
    }
    else
    {
        throw ngraph_error("Invalid request for vector zero of " + dtype + std::to_string(n));
    }
    return ss.str();
}
#define LOAD_C                                                                                     \
    "__device__ __forceinline__ " << dtype << " " << name << "(const " << dtype                    \
                                  << "*  __restrict__ in, int i=0, bool b=true) { " << dtype       \
                                  << "  v; zero_(v); if (b) v = in[i]; return v; }\n"

std::string ngraph::runtime::gpu::nvrtc::define_coherent_load(const std::string& dtype,
                                                              const std::string& name)
{
    std::stringstream ss;
    ss << define_zero(dtype);
    ss << LOAD_C;
    return ss.str();
}

std::string ngraph::runtime::gpu::nvrtc::define_coherent_vload(const std::string& dtype,
                                                               const uint32_t& n,
                                                               const std::string& name)
{
    std::stringstream ss;
    ss << define_vzero(dtype, n);
    ss << LOAD_C;
    return ss.str();
}

#define LOAD_NC                                                                                    \
    "__device__ __forceinline__ " << dtype << " " << name << "(const " << dtype                    \
                                  << "*  __restrict__ in, int i=0, bool b=true) { " << dtype       \
                                  << "  v; zero_(v); if (b) v = __ldg(in + i); return v; }\n"

std::string ngraph::runtime::gpu::nvrtc::define_non_coherent_load(const std::string& dtype,
                                                                  const std::string& name)
{
    std::stringstream ss;
    ss << define_zero(dtype);
    ss << LOAD_NC;
    return ss.str();
}

std::string ngraph::runtime::gpu::nvrtc::define_non_coherent_vload(const std::string& dtype,
                                                                   const uint32_t& n,
                                                                   const std::string& name)
{
    std::stringstream ss;
    ss << define_vzero(dtype, n);
    ss << LOAD_NC;
    return ss.str();
}
