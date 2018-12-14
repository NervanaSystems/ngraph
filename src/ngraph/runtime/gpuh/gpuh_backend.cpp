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

#include "ngraph/runtime/gpuh/gpuh_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace ngraph;
using namespace std;

// #define DEBUG_MODE_INT

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::gpuh::GPUHBackend();
}

vector<string> get_excludes()
{
    return vector<string>{{"Not"}};
}

#ifdef DEBUG_MODE_INT
runtime::gpuh::GPUHBackend::GPUHBackend()
    : HybridBackend({make_shared<ngraph::runtime::interpreter::INTBackend>(get_excludes()),
                     make_shared<ngraph::runtime::interpreter::INTBackend>()})
{
}
#else
runtime::gpuh::GPUHBackend::GPUHBackend()
    : HybridBackend({make_shared<ngraph::runtime::gpu::GPU_Backend>(),
                     make_shared<ngraph::runtime::interpreter::INTBackend>()})
{
}
#endif
