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
#include <algorithm>
#include <map>

#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/util.hpp"



#define NVRTC_SAFE_CALL(x) \ 
do { \ 
    nvrtcResult result = x; \ 
        if (result != NVRTC_SUCCESS) { \ 
            std::cerr << "\nerror: " #x " failed with error " \ 
                << nvrtcGetErrorString(result) << '\n'; \
                exit(1); \ 
        } \
} while(0) 

#define CUDA_SAFE_CALL(x) \ 
do { \ 
    CUresult result = x; \ 
        if (result != CUDA_SUCCESS) { \ 
            const char *msg; \ 
                cuGetErrorName(result, &msg); \ 
                std::cerr << "\nerror: " #x " failed with error " \ 
                << msg << '\n'; \ 
                exit(1); \ 
        } \ 
} while(0)

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace cuda
            {
                namespace kernel
                {

                    void emit_abs(void* in, void* out, size_t count)
                    {
                            const char *op_abs = R"(  
  extern "C" __global__  
  void cuda_op_abs(float* in, float* out, size_t n)  
  {  
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
    if(tid < n) 
    {
      out[tid] = fabsf(in[tid]);  
    }
  })";

    // Create an instance of nvrtcProgram with the code string. 

    nvrtcProgram prog; 
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog i
                op_abs, // buffer 
                "op_abs.cu", // name 
                0, // numHeaders 
                NULL, // headers 
                NULL)); // includeNames


    const char *opts[] = {"--gpu-architecture=compute_35",
        "--relocatable-device-code=true"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog 
            2, // numOptions 
            opts); // options
    // Obtain compilation log from the program. 

    size_t logSize; 

    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize)); 
    char *log = new char[logSize]; 
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log)); 
    std::cout << log << '\n'; 
    delete[] log; 

    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }

    size_t ptxSize; 
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize)); 
    char *ptx = new char[ptxSize]; 
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx)); // Destroy the program. 
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); // Load the generated PTX and get a handle to the parent kernel. 

      CUdevice cuDevice;
      CUcontext context;
      CUmodule module;
      CUfunction cuda_op_abs_kernel;
      CUDA_SAFE_CALL( cuInit(0));
      CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
      CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice)); 
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&cuda_op_abs_kernel, module, "cuda_op_abs"));
 
    size_t numBlocks = 4;
    size_t numThreads = 4; 
    size_t nt = numBlocks * numThreads; 
    size_t bufferSize = nt * sizeof(float); 
    float *hOut = new float[nt]; 
    float *hIn = new float[nt]; 
    for(int i = 0; i< nt; i++) hIn[i] = -i;
    
//    void *dOut, *dIn;
//    cudaMalloc((void**) &dIn, 64);
//    cudaMalloc((void**) &dOut, 64);   
    CUdeviceptr dPtrIn, dPtrOut;
    dPtrIn = (CUdeviceptr)in;
    dPtrOut = (CUdeviceptr)out;
    
                        void *argsList[] = {&dPtrIn, &dPtrOut, &nt};
  //  cudaLaunchKernel(cuda_op_obs_kernel,
  //                   {4, 1, 1},
  //                   {1, 1, 1},
  //                    argslist, 0, NULL);
 
                 //       void *argsList[] = {dIn, dOut, &nt};
                        CUDA_SAFE_CALL(
                        cuLaunchKernel(cuda_op_abs_kernel, 
                        4 , 1, 1, // grid dim 
                        4, 1, 1, // block dim 
                        0, NULL, // shared mem and stream 
                        argsList, 0)); // arguments 
                        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output. 
                    }

                    void emit_broadcast(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& broadcast_axes)
                    {
                    }

                    //
                    // For the reference kernel this is gpud on, see ngraph/runtime/kernel/concat.hpp.
                    //
                    void emit_concat(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::vector<std::string>& args,
                            const std::string& out,
                            const std::vector<Shape>& in_shapes,
                            const Shape& out_shape,
                            size_t concatenation_axis)
                    {
                    }

                    void emit_replace_slice(
                            codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& arg1, // replacement value
                            const std::string& out,
                            const Shape& arg1_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_slice(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_reshape(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisVector& arg0_axis_order)
                    {
                    }

                    void emit_sum(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& reduction_axes)
                    {
                    }

                }
            }
        }
    }
}
