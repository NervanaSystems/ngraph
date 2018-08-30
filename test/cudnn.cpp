//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"

#include "ngraph/ngraph.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(cudnn, loadTest)
{
    auto cudnn_version = cudnnGetVersion();
    EXPECT_FLOAT_EQ(cudnn_version, CUDNN_VERSION);
}

TEST(cudnn, compileTest)
{
    const auto source = R"###(
// Example developed from LLVM documentation https://llvm.org/docs/NVPTXUsage.html

#include <cassert>
#include <fstream>
#include <iostream>
#include "cublas_v2.h"
#include "cuda.h"

void check_cuda_errors(CUresult err) {
  assert(err == CUDA_SUCCESS);
}

/// main - Program entry point
int main(int argc, char **argv) {
  CUdevice    device;
  CUmodule    cuda_module;
  CUcontext   context;
  CUfunction  function;
  CUlinkState linker;
  int         dev_count;

  // Cublas init

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  cublasDestroy(handle);

  // CUDA initialization
  check_cuda_errors(cuInit(0));
  check_cuda_errors(cuDeviceGetCount(&dev_count));
  check_cuda_errors(cuDeviceGet(&device, 0));

  char name[128];
  check_cuda_errors(cuDeviceGetName(name, 128, device));
  std::cout << "Using CUDA Device [0]: " << name << "\n";

  int dev_major, dev_minor;
  check_cuda_errors(cuDeviceComputeCapability(&dev_major, &dev_minor, device));
  std::cout << "Device Compute Capability: "
            << dev_major << "." << dev_minor << "\n";
  if (dev_major < 2) {
    std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    return 1;
  }

const auto str = R"(
  .version 5.0
  .target sm_60
  .address_size 64

    // .globl	_Z7ew_multPfS_S_ // -- Begin function _Z7ew_multPfS_S_
  .global .align 1 .b8 threadIdx[1];
                                          // @_Z7ew_multPfS_S_
  .visible .entry _Z7ew_multPfS_S_(
    .param .u64 _Z7ew_multPfS_S__param_0,
    .param .u64 _Z7ew_multPfS_S__param_1,
    .param .u64 _Z7ew_multPfS_S__param_2
  )
  {
    .local .align 8 .b8 	__local_depot0[24];
    .reg .b64 	%SP;
    .reg .b64 	%SPL;
    .reg .f32 	%f<4>;
    .reg .b32 	%r<2>;
    .reg .b64 	%rd<17>;

  // BB#0:
    mov.u64 	%SPL, __local_depot0;
    cvta.local.u64 	%SP, %SPL;
    ld.param.u64 	%rd3, [_Z7ew_multPfS_S__param_2];
    ld.param.u64 	%rd2, [_Z7ew_multPfS_S__param_1];
    ld.param.u64 	%rd1, [_Z7ew_multPfS_S__param_0];
    cvta.to.global.u64 	%rd4, %rd3;
    cvta.global.u64 	%rd5, %rd4;
    cvta.to.global.u64 	%rd6, %rd2;
    cvta.global.u64 	%rd7, %rd6;
    cvta.to.global.u64 	%rd8, %rd1;
    cvta.global.u64 	%rd9, %rd8;
    st.u64 	[%SP+0], %rd9;
    st.u64 	[%SP+8], %rd7;
    st.u64 	[%SP+16], %rd5;
    ld.u64 	%rd10, [%SP+0];
    mov.u32 	%r1, %tid.x;
    mul.wide.u32 	%rd11, %r1, 4;
    add.s64 	%rd12, %rd10, %rd11;
    ld.f32 	%f1, [%rd12];
    ld.u64 	%rd13, [%SP+8];
    add.s64 	%rd14, %rd13, %rd11;
    ld.f32 	%f2, [%rd14];
    mul.rn.f32 	%f3, %f1, %f2;
    ld.u64 	%rd15, [%SP+16];
    add.s64 	%rd16, %rd15, %rd11;
    st.f32 	[%rd16], %f3;
    ret;
  }
                                          // -- End function
    // .globl	_Z6ew_addPfS_S_ // -- Begin function _Z6ew_addPfS_S_
  .visible .entry _Z6ew_addPfS_S_(
    .param .u64 _Z6ew_addPfS_S__param_0,
    .param .u64 _Z6ew_addPfS_S__param_1,
    .param .u64 _Z6ew_addPfS_S__param_2
  )                                       // @_Z6ew_addPfS_S_
  {
    .local .align 8 .b8 	__local_depot1[24];
    .reg .b64 	%SP;
    .reg .b64 	%SPL;
    .reg .f32 	%f<4>;
    .reg .b32 	%r<2>;
    .reg .b64 	%rd<17>;

  // BB#0:
    mov.u64 	%SPL, __local_depot1;
    cvta.local.u64 	%SP, %SPL;
    ld.param.u64 	%rd3, [_Z6ew_addPfS_S__param_2];
    ld.param.u64 	%rd2, [_Z6ew_addPfS_S__param_1];
    ld.param.u64 	%rd1, [_Z6ew_addPfS_S__param_0];
    cvta.to.global.u64 	%rd4, %rd3;
    cvta.global.u64 	%rd5, %rd4;
    cvta.to.global.u64 	%rd6, %rd2;
    cvta.global.u64 	%rd7, %rd6;
    cvta.to.global.u64 	%rd8, %rd1;
    cvta.global.u64 	%rd9, %rd8;
    st.u64 	[%SP+0], %rd9;
    st.u64 	[%SP+8], %rd7;
    st.u64 	[%SP+16], %rd5;
    ld.u64 	%rd10, [%SP+0];
    mov.u32 	%r1, %tid.x;
    mul.wide.u32 	%rd11, %r1, 4;
    add.s64 	%rd12, %rd10, %rd11;
    ld.f32 	%f1, [%rd12];
    ld.u64 	%rd13, [%SP+8];
    add.s64 	%rd14, %rd13, %rd11;
    ld.f32 	%f2, [%rd14];
    add.rn.f32 	%f3, %f1, %f2;
    ld.u64 	%rd15, [%SP+16];
    add.s64 	%rd16, %rd15, %rd11;
    st.f32 	[%rd16], %f3;
    ret;
  }
                                          // -- End function
)";

  // Create driver context
  check_cuda_errors(cuCtxCreate(&context, 0, device));

  // Create module for object
  check_cuda_errors(cuModuleLoadDataEx(&cuda_module, str, 0, 0, 0));

  // Get kernel function
  check_cuda_errors(cuModuleGetFunction(&function, cuda_module, "_Z7ew_multPfS_S_"));

  // Device data
  CUdeviceptr dev_bufferA;
  CUdeviceptr dev_bufferB;
  CUdeviceptr dev_bufferC;

  check_cuda_errors(cuMemAlloc(&dev_bufferA, sizeof(float)*16));
  check_cuda_errors(cuMemAlloc(&dev_bufferB, sizeof(float)*16));
  check_cuda_errors(cuMemAlloc(&dev_bufferC, sizeof(float)*16));

  float* host_A = new float[16];
  float* host_B = new float[16];
  float* host_C = new float[16];

  // Populate input
  for (unsigned i = 0; i != 16; ++i) {
    host_A[i] = (float)i;
    host_B[i] = (float)(2*i);
    host_C[i] = 0.0f;
  }

  check_cuda_errors(cuMemcpyHtoD(dev_bufferA, &host_A[0], sizeof(float)*16));
  check_cuda_errors(cuMemcpyHtoD(dev_bufferB, &host_B[0], sizeof(float)*16));

  unsigned block_size_X = 16;
  unsigned block_size_Y = 1;
  unsigned block_size_Z = 1;
  unsigned grid_size_X  = 1;
  unsigned grid_size_Y  = 1;
  unsigned grid_size_Z  = 1;

  // Kernel parameters
  void *kernel_params[] = { &dev_bufferA, &dev_bufferB, &dev_bufferC };

  std::cout << "Launching kernel\n";

  // Kernel launch
  check_cuda_errors(cuLaunchKernel(function, grid_size_X, grid_size_Y, grid_size_Z,
                                 block_size_X, block_size_Y, block_size_Z,
                                 0, NULL, kernel_params, NULL));

  // Retrieve device data
  check_cuda_errors(cuMemcpyDtoH(&host_C[0], dev_bufferC, sizeof(float)*16));

  std::cout << "Results:\n";
  for (unsigned i = 0; i != 16; ++i) {
    std::cout << host_A[i] << " + " << host_B[i] << " = " << host_C[i] << "\n";
  }

  // Clean up after ourselves
  delete [] host_A;
  delete [] host_B;
  delete [] host_C;

  // Clean-up
  check_cuda_errors(cuMemFree(dev_bufferA));
  check_cuda_errors(cuMemFree(dev_bufferB));
  check_cuda_errors(cuMemFree(dev_bufferC));
  check_cuda_errors(cuModuleUnload(cuda_module));
  check_cuda_errors(cuCtxDestroy(context));

  return 0;
})###";
    codegen::Compiler compiler;

    auto module = compiler.compile(source);
}
