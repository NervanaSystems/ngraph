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

#include <string>

#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

extern "C" void ngraph::runtime::gpu::invoke_primitive(const GPURuntimeContext* ctx,
                                                       size_t primitive_index,
                                                       void** args,
                                                       void** result)
{
    (*ctx->gpu_primitives[primitive_index])(args, result);
}

extern "C" void* ngraph::runtime::gpu::invoke_memory_primitive(const GPURuntimeContext* ctx,
                                                               size_t primitive_index)
{
    return ctx->gpu_memory_primitives[primitive_index]();
}
