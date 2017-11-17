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

#include "ngraph/codegen/nvptx_compiler.hpp"
#include "ngraph/codegen/nvptx_execution_engine.hpp"

#include "ngraph/runtime/gpu/call_frame.hpp"
#include "ngraph/runtime/gpu/backend.hpp"
#include "ngraph/runtime/gpu/external_function.hpp"
#include "ngraph/runtime/gpu/manager.hpp"

using namespace ngraph::runtime::gpu;
using namespace std;

TEST(cudnn, loadTest)
{
    auto cudnn_version = cudnnGetVersion();
    EXPECT_FLOAT_EQ(cudnn_version, CUDNN_VERSION);
}

TEST(cudnn, loadBackend)
{
  // auto gpu_call_frame = new GPUCallFrame();
  // auto gpu_backend = new GPUBackend();
  // auto gpu_external_function = new GPUExternalFunction();
  // auto gpu_manager = new GPUManager();
}

TEST(cudnn, compileTest)
{
  // constexpr auto name = "test.cpp";
  const auto source = R"(
  #include <iostream>

    __global__ void axpy(float a, float* x, float* y) {
      y[threadIdx.x] = a * x[threadIdx.x];
    }

    int main(int argc, char* argv[]) {
      const int kDataLen = 4;

      float a = 2.0f;
      float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
      float host_y[kDataLen];

      // Copy input data to device.
      float* device_x;
      float* device_y;
      cudaMalloc(&device_x, kDataLen * sizeof(float));
      cudaMalloc(&device_y, kDataLen * sizeof(float));
      cudaMemcpy(device_x, host_x, kDataLen * sizeof(float),
                cudaMemcpyHostToDevice);

      // Launch the kernel.
      axpy<<<1, kDataLen>>>(a, device_x, device_y);

      // Copy output data to host.
      cudaDeviceSynchronize();
      cudaMemcpy(host_y, device_y, kDataLen * sizeof(float),
                cudaMemcpyDeviceToHost);

      // Print the results.
      for (int i = 0; i < kDataLen; ++i) {
        std::cout << "y[" << i << "] = " << host_y[i] << "\n";
      }

      cudaDeviceReset();
      return 0;
    })";
  EXPECT_EQ(source,source);
}
