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

#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include "ngraph/runtime/gpu/kernels/batchnorm.hpp"
#include "ngraph/runtime/gpu/kernels/helpers.hpp"

using namespace ngraph;

// batch normalization inference
// y = g * (x - mean) / sqrt(var + eps) + b
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) batchnorm_inference_ncdhw(
    T*              Y,
    const float* __restrict__ M,
    const float* __restrict__ V,
    const     T* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ B,
    int CDHW, int DHW, float epsilon)
{
    const int tid = threadIdx.x;
    const int c   = blockIdx.x;
    const int n   = blockIdx.y;

    int offset = n * CDHW + c * DHW;

    float g = G[c];
    float b = B[c];

    float mean = M[c];
    float var  = V[c];

    float rstdg = rsqrtf(var + epsilon) * g;

    X += offset;
    Y += offset;
    for (int i = tid; i < DHW; i += THREADS)
    {
        float x = load(X, i);
        float y = (x - mean) * rstdg + b;
        store(Y, y, i);
    }
}

template <typename T>
bool runtime::gpu::BatchNormNCDHW_Inference(T* y,
                                            const float* m,
                                            const float* v,
                                            const     T* x,
                                            const float* g,
                                            const float* b,
                                            int N, int C, int DHW, float epsilon)
{
    int CDHW = C*DHW;
    dim3 grid(C, N, 1);
    if (DHW < 128*8)
    {
        batchnorm_inference_ncdhw<T, 32><<<grid,  32, 0>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    }
    else if (DHW < 512*8)
    {
        batchnorm_inference_ncdhw<T,128><<<grid, 128, 0>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    }
    else
    {
        batchnorm_inference_ncdhw<T,512><<<grid, 512, 0>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    }
    return true;
}
template bool runtime::gpu::BatchNormNCDHW_Inference<float>(float* y,
                                                            const float* m,
                                                            const float* v,
                                                            const float* x,
                                                            const float* g,
                                                            const float* b,
                                                            int N, int C, int DHW, float epsilon);
