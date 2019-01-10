//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/graph_util.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_executable.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::GPUExecutable::GPUExecutable(shared_ptr<BackendContext> context,
                                           shared_ptr<Function> func,
                                           bool enable_performance_collection)
    : m_context(context)
{
    m_context->bind_cuda_context_to_thread();
    m_external_function = make_shared<GPU_ExternalFunction>(func, m_context);
    m_external_function->m_emit_timing = m_performance_counters_enabled;
    m_external_function->compile();
    m_compiled_function = m_external_function->m_compiled_function;
    m_inputs.resize(func->get_parameters().size());
    m_outputs.resize(func->get_output_size());

    set_parameters_and_results(*func);
}

void runtime::gpu::GPUExecutable::initialize_io(void** target,
                                                const vector<shared_ptr<runtime::Tensor>>& source)
{
    for (size_t i = 0; i < source.size(); i++)
    {
        shared_ptr<runtime::gpu::GPUTensor> tensor =
            dynamic_pointer_cast<runtime::gpu::GPUTensor>(source[i]);
        if (tensor)
        {
            target[i] = tensor->m_allocated_buffer_pool;
        }
        else
        {
            throw invalid_argument("Tensors passed to GPU backend must be GPU Tensors");
        }
    }
}

bool runtime::gpu::GPUExecutable::execute(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // ensure the GPURuntimeContext primitive pointers are valid
    m_context->prepare_runtime_context();

    // Device tensors
    initialize_io(m_inputs.data(), inputs);
    initialize_io(m_outputs.data(), outputs);

    auto ctx = m_context->m_runtime_context.get();
    m_compiled_function(m_inputs.data(), m_outputs.data(), ctx);

    return true;
}

vector<runtime::PerformanceCounter> runtime::gpu::GPUExecutable::get_performance_data() const
{
    std::vector<runtime::PerformanceCounter> rc;
    if (m_external_function != nullptr)
    {
        auto* engine = m_external_function->m_execution_engine.get();
        if (engine)
        {
            auto get_count = engine->find_function<size_t()>("get_debug_timer_count");
            auto get_name = engine->find_function<const char*(size_t)>("get_debug_timer_name");
            auto get_microseconds =
                engine->find_function<size_t(size_t)>("get_debug_timer_microseconds");
            auto get_call_count =
                engine->find_function<size_t(size_t)>("get_debug_timer_call_count");

            if (get_count && get_name && get_microseconds && get_call_count)
            {
                size_t count = get_count();
                for (size_t i = 0; i < count; i++)
                {
                    rc.push_back({get_name(i), get_microseconds(i), get_call_count(i)});
                }
            }
        }
    }
    return rc;
}
