//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <cstdlib>
#include <string>
#include <tuple>
#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/graph_comparison.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

using compare_gpu_cpu = ngraph::model_comparison<GPU, CPU>;

// The set of graphs tested is not currently significant. These graphs were
// chosen because they're already availabe and demonstrate the technique.
NGRAPH_COMPARISON_TEST(
    tf_resnet8_files,
    compare_gpu_cpu,
    testing::Values("tensorflow/resnet8/"
                    "tf_function_cluster_12[_XlaCompiledKernel=true,_XlaNumConstantArgs=3,_"
                    "XlaNumResourceArgs=0].v23.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_20[_XlaCompiledKernel=true,_XlaNumConstantArgs=3,_"
                    "XlaNumResourceArgs=0].v23.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_22[_XlaCompiledKernel=true,_XlaNumConstantArgs=4,_"
                    "XlaNumResourceArgs=0].v24.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_23[_XlaCompiledKernel=true,_XlaNumConstantArgs=1,_"
                    "XlaNumResourceArgs=0].v296.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_28[_XlaCompiledKernel=true,_XlaNumConstantArgs=0,_"
                    "XlaNumResourceArgs=0].v13.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_4[_XlaCompiledKernel=true,_XlaNumConstantArgs=1,_"
                    "XlaNumResourceArgs=0].v14.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_8[_XlaCompiledKernel=true,_XlaNumConstantArgs=2,_"
                    "XlaNumResourceArgs=0].v28.json"));
