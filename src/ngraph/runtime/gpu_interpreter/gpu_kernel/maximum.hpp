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
#include <cublas_v2.h>
#include <cudnn.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

namespace ngraph
{
    namespace runtime
    {
        namespace gpu_kernel
        {
            template <typename T>
            void maximum(T* arg0, T* arg1, T* out, size_t count)
            {
                 for (size_t i = 0; i < count; i++)
                 {
                     out[i] = arg0[i] > arg1[i] ? arg0[i] : arg1[i];
                 }
            }
	    
	    template<>
	    inline void maximum(float* arg0, float* arg1, float* out, size_t count)
	    {
		cudnnHandle_t cudnnHandle;
		checkCUDNN(cudnnCreate(&cudnnHandle));
		
		cudnnTensorDescriptor_t descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/count));


		cudnnOpTensorDescriptor_t opTensorDesc;
		checkCUDNN(cudnnCreateOpTensorDescriptor(&opTensorDesc));

		checkCUDNN(cudnnSetOpTensorDescriptor(opTensorDesc,
                                      CUDNN_OP_TENSOR_MAX,
				      CUDNN_DATA_FLOAT,
				      CUDNN_NOT_PROPAGATE_NAN));

		float* d_arg0;
		float* d_arg1;
		float* d_out;

	 	cudaMalloc((void**) &d_arg0, sizeof(float) * count);
	 	cudaMalloc((void**) &d_arg1, sizeof(float) * count);
	 	cudaMalloc((void**) &d_out, sizeof(float) * count);
		cudaMemcpy(d_arg0, (float *)arg0, sizeof(float) * count, cudaMemcpyHostToDevice);	
		cudaMemcpy(d_arg1, (float *)arg1, sizeof(float) * count, cudaMemcpyHostToDevice);
		
		float alpha1 = 1.0, alpha2 = 1.0, beta = 0;	
		checkCUDNN(cudnnOpTensor( cudnnHandle, 
			 opTensorDesc, 
			 &alpha1, 
			 descriptor, 
			 d_arg0,
			 &alpha2, 
			 descriptor, 
			 d_arg1,
			 &beta, 
			 descriptor, 
			 d_out));

		cudaMemcpy(out, d_out, sizeof(float) * count, cudaMemcpyDeviceToHost);
		cudaFree(d_arg0);
		cudaFree(d_arg1);
		cudaFree(d_out);	
 		cudnnDestroy(cudnnHandle);
            }
        }
    }
}
