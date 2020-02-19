//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/gpu/gpu_executable.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_internal_function.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::GPU_Executable::GPU_Executable(shared_ptr<Function> func, bool enable_timing)
    : m_context(new GPU_Backend::BackendContext())

{
    FunctionInstance& instance = m_function_instance;
    if (instance.m_compiled_function == nullptr)
    {
        m_context->bind_cuda_context_to_thread();
        instance.m_compiled_function = runtime::gpu::GPUCompiledFunction::make(func, m_context);
        instance.m_compiled_function->m_emit_timing = enable_timing;
        instance.m_compiled_function->compile();
        instance.m_runtime = instance.m_compiled_function->m_runtime;
        instance.m_inputs.resize(func->get_parameters().size());
        instance.m_outputs.resize(func->get_output_size());
    }
    set_parameters_and_results(*func);
}

void runtime::gpu::GPU_Executable::initialize_io(void** target,
                                                 const vector<shared_ptr<runtime::Tensor>>& source)
{
    for (size_t i = 0; i < source.size(); i++)
    {
        shared_ptr<runtime::gpu::GPUTensor> tv =
            dynamic_pointer_cast<runtime::gpu::GPUTensor>(source[i]);
        if (tv)
        {
            target[i] = tv->m_allocated_buffer_pool;
        }
        else
        {
            throw invalid_argument("Tensors passed to GPU backend must be GPU Tensors");
        }
    }
}

bool runtime::gpu::GPU_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    FunctionInstance& instance = m_function_instance;
    if (instance.m_compiled_function == nullptr)
    {
        throw runtime_error("compile() must be called before call().");
    }

    // ensure the GPURuntimeContext primitive pointers are valid
    m_context->prepare_runtime_context();

    // Device tensors
    initialize_io(instance.m_inputs.data(), inputs);
    initialize_io(instance.m_outputs.data(), outputs);

    auto ctx = m_context->m_runtime_context.get();
    instance.m_runtime(instance.m_inputs.data(), instance.m_outputs.data(), ctx);

    return true;
}

vector<runtime::PerformanceCounter> runtime::gpu::GPU_Executable::get_performance_data() const
{
    std::vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_instance;
    if (instance.m_compiled_function != nullptr)
    {
        instance.m_compiled_function->get_performance_data(rc);
    }
    return rc;
}
