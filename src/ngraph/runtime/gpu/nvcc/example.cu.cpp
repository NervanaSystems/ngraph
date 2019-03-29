//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <iostream>
#include "ngraph/runtime/gpu/nvcc/kernels.hpp"
using namespace ngraph;

__global__ void example()
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from tid = %d\n", tid);
    __syncthreads();
}

void runtime::gpu::example_kernel()
{
    example<<<1, 32>>>();
    return;
}
