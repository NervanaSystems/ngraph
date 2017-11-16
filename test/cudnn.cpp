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

#include "ngraph/runtime/gpu/call_frame.hpp"
#include "ngraph/runtime/gpu/backend.hpp"
#include "ngraph/runtime/gpu/external_function.hpp"
#include "ngraph/runtime/gpu/manager.hpp"

using namespace ngraph::runtime::gpu;

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
