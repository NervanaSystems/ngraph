// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#ifndef _CUDA_VEC_ABS_
#define _CUDA_VEC_ABS_

#include "abs.hpp"
#include <cuda_runtime.h>

__global__ void VecAbs(float* A, float* B) 
{ 
    int i = threadIdx.x; 
    B[i] = A[i] < 0 ? -A[i] : A[i]; 
} 

extern "C"
void runVecAbs(float* arg, float* out, size_t count)
{
	float *d_arg, *d_out;
	
	cudaMalloc((void **)& d_arg, sizeof(float) * count);
	cudaMalloc((void **)& d_out, sizeof(float) * count);
  	
	cudaMemcpy(d_arg, arg, count, cudaMemcpyHostToDevice);
	
        VecAbs<<<1, count>>>(d_arg, d_out);

	cudaMemcpy(out, d_out, count, cudaMemcpyDeviceToHost);
	
	cudaFree(d_arg);
	cudaFree(d_out);
}
#endif
