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

#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cudnn.h>

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/ngraph.hpp"

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
#include <cassert>
#include <fstream>
#include <iostream>
#include "cuda.h"


void checkCudaErrors(CUresult err) {
  assert(err == CUDA_SUCCESS);
}

/// main - Program entry point
int main(int argc, char **argv) {
  CUdevice    device;
  CUmodule    cudaModule;
  CUcontext   context;
  CUfunction  function;
  CUlinkState linker;
  int         devCount;

  // CUDA initialization
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&devCount));
  checkCudaErrors(cuDeviceGet(&device, 0));

  char name[128];
  checkCudaErrors(cuDeviceGetName(name, 128, device));
  std::cout << "Using CUDA Device [0]: " << name << "\n";

  int devMajor, devMinor;
  checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
  std::cout << "Device Compute Capability: "
            << devMajor << "." << devMinor << "\n";
  if (devMajor < 2) {
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
  checkCudaErrors(cuCtxCreate(&context, 0, device));

  // Create module for object
  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str, 0, 0, 0));

  // Get kernel function
  checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "_Z7ew_multPfS_S_"));

  // Device data
  CUdeviceptr devBufferA;
  CUdeviceptr devBufferB;
  CUdeviceptr devBufferC;

  checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(float)*16));
  checkCudaErrors(cuMemAlloc(&devBufferB, sizeof(float)*16));
  checkCudaErrors(cuMemAlloc(&devBufferC, sizeof(float)*16));

  float* hostA = new float[16];
  float* hostB = new float[16];
  float* hostC = new float[16];

  // Populate input
  for (unsigned i = 0; i != 16; ++i) {
    hostA[i] = (float)i;
    hostB[i] = (float)(2*i);
    hostC[i] = 0.0f;
  }

  checkCudaErrors(cuMemcpyHtoD(devBufferA, &hostA[0], sizeof(float)*16));
  checkCudaErrors(cuMemcpyHtoD(devBufferB, &hostB[0], sizeof(float)*16));


  unsigned blockSizeX = 16;
  unsigned blockSizeY = 1;
  unsigned blockSizeZ = 1;
  unsigned gridSizeX  = 1;
  unsigned gridSizeY  = 1;
  unsigned gridSizeZ  = 1;

  // Kernel parameters
  void *KernelParams[] = { &devBufferA, &devBufferB, &devBufferC };

  std::cout << "Launching kernel\n";

  // Kernel launch
  checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                 blockSizeX, blockSizeY, blockSizeZ,
                                 0, NULL, KernelParams, NULL));

  // Retrieve device data
  checkCudaErrors(cuMemcpyDtoH(&hostC[0], devBufferC, sizeof(float)*16));


  std::cout << "Results:\n";
  for (unsigned i = 0; i != 16; ++i) {
    std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << "\n";
  }


  // Clean up after ourselves
  delete [] hostA;
  delete [] hostB;
  delete [] hostC;

  // Clean-up
  checkCudaErrors(cuMemFree(devBufferA));
  checkCudaErrors(cuMemFree(devBufferB));
  checkCudaErrors(cuMemFree(devBufferC));
  checkCudaErrors(cuModuleUnload(cudaModule));
  checkCudaErrors(cuCtxDestroy(context));

  return 0;
})###";
    codegen::Compiler compiler;

    auto module = compiler.compile(source);
}

TEST(cudnn, abc)
{
    using f32 = element::Float32;

    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(f32::element_type(), shape);
    auto B = make_shared<op::Parameter>(f32::element_type(), shape);
    auto C = make_shared<op::Parameter>(f32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(f32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("GPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
}
