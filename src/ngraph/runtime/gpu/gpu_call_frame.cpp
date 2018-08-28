/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdlib>
#include <fstream>
#include <stdio.h>

#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace std;
using namespace ngraph;

runtime::gpu::GPU_CallFrame::GPU_CallFrame(std::shared_ptr<GPU_ExternalFunction> external_function,
                                           EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_function(compiled_function)
{
}

runtime::gpu::GPU_CallFrame::~GPU_CallFrame()
{
}

void runtime::gpu::GPU_CallFrame::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::TensorView>>& input_tvs,
    GPURuntimeContext* ctx)
{
    // Device tensors
    vector<void*> inputs;
    vector<void*> outputs;

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::gpu::GPU_TensorView> tv =
            static_pointer_cast<runtime::gpu::GPU_TensorView>(input_tvs[i]);
        inputs.push_back(tv->m_allocated_buffer_pool);
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::gpu::GPU_TensorView> tv =
            static_pointer_cast<runtime::gpu::GPU_TensorView>(output_tvs[i]);
        outputs.push_back(tv->m_allocated_buffer_pool);
    }

    m_compiled_function(inputs.data(), outputs.data(), ctx);
}
