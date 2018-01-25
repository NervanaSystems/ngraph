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

#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h" 

namespace ngraph
{
    namespace runtime
    {
        namespace gpu_kernel
        {
	    	template<typename T>
            void add(T* arg0, T* arg1, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = arg0[i] + arg1[i];
                }
            }

	    	template<>
            inline void add<float>(float* arg0, float* arg1, float* out, size_t count)
            {
				float* d_arg0;
				float* d_arg1;
				float* d_out;
				cudaMalloc((void**) &d_arg0, sizeof(float) * count);
				cudaMalloc((void**) &d_arg1, sizeof(float) * count);
				cudaMalloc((void**) &d_out, sizeof(float) * count);

				cudaMemcpy(d_arg0, (float *)arg0, sizeof(float) * count, cudaMemcpyHostToDevice);
				cudaMemcpy(d_arg1, (float *)arg1, sizeof(float) * count, cudaMemcpyHostToDevice);

				cublasStatus_t ret;  
				cublasHandle_t handle;
				ret = cublasCreate(&handle);
					
				float alpha = 1.0;
				float beta = 1.0;
				ret = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, 1, 
							&alpha, d_arg0, count, 
							&beta, d_arg1, count,
							d_out, count);

				cudaMemcpy((float*) out, d_out, sizeof(float) * count, cudaMemcpyDeviceToHost);
				cublasDestroy(handle);
            }
        }
    }
}
