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

__device__ __forceinline__ void ew_zero(float  &a) { a = 0.0f; }

__device__ __forceinline__ void __stg(const float *ptr, float val)
{
    asm volatile ("st.global.wb.f32 [%0], %1;" :: "l"(ptr), "f"(val)  );
}

__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true) { float  v; ew_zero(v); if (b) v = __ldg(in + i); return v; }

__device__ __forceinline__ void store(float*  out, float  v, int i=0, bool b=true) { if (b) __stg(out + i, v); }

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
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
        "}"
        : "=r"(result)
        : "r"(value), "r"(magic), "r"(shift));
    return result;
}
