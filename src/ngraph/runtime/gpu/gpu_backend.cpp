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

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::gpu::GPU_Backend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::gpu::GPU_Backend::GPU_Backend()
    : runtime::Backend()
    , m_context(new BackendContext())
{
}

runtime::gpu::GPU_Backend::BackendContext::BackendContext()
    : m_runtime_context(new GPURuntimeContext)
    , m_primitive_emitter(new GPUPrimitiveEmitter(m_runtime_context))
    , m_cuda_manager(new CudaContextManager)
{
    // Create context use driver API and make it current, the runtime call will pickup the context
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    // #interoperability-between-runtime-and-driver-apis
    bind_cuda_context_to_thread();

    m_runtime_context->cublas_handle = new cublasHandle_t;
    cublasStatus_t cublasStatus = cublasCreate(m_runtime_context->cublas_handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        throw runtime_error("cuBLAS create handle failed");
    }
    // Pass scalars as reference on the Device
    cublasSetPointerMode(*m_runtime_context->cublas_handle, CUBLAS_POINTER_MODE_DEVICE);

    m_runtime_context->cudnn_handle = new cudnnHandle_t;
    cudnnStatus_t cudnnStatus = cudnnCreate(m_runtime_context->cudnn_handle);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS)
    {
        throw runtime_error("cuDNN create handle failed");
    }

    // register with c-api runtime context
    m_runtime_context->compiled_kernel_pool = new CudaFunctionPool;
}

void runtime::gpu::GPU_Backend::BackendContext::prepare_runtime_context()
{
    // set context current each time in case thread changed
    bind_cuda_context_to_thread();
    // add pointers to gpu primitives into the gpu runtime context
    m_runtime_context->gpu_primitives = m_primitive_emitter->get_primitives().data();
    m_runtime_context->gpu_memory_primitives = m_primitive_emitter->get_memory_primitives().data();
}

void runtime::gpu::GPU_Backend::BackendContext::bind_cuda_context_to_thread()
{
    m_cuda_manager->SetContextCurrent();
}

runtime::gpu::GPU_Backend::BackendContext::~BackendContext()
{
    cublasDestroy(*m_runtime_context->cublas_handle);
    delete m_runtime_context->cublas_handle;
    cudnnDestroy(*m_runtime_context->cudnn_handle);
    delete m_runtime_context->cudnn_handle;
    delete m_runtime_context->compiled_kernel_pool;
}

shared_ptr<runtime::Tensor>
    runtime::gpu::GPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::gpu::GPUTensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::gpu::GPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::gpu::GPUTensor>(element_type, shape, memory_pointer);
}

runtime::Handle runtime::gpu::GPU_Backend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        m_context->bind_cuda_context_to_thread();
        instance.m_external_function = make_shared<GPU_ExternalFunction>(func, m_context);
        instance.m_external_function->m_emit_timing = instance.m_performance_counters_enabled;
        instance.m_external_function->compile();
        instance.m_compiled_function = instance.m_external_function->m_compiled_function;
        instance.m_inputs.resize(func->get_parameters().size());
        instance.m_outputs.resize(func->get_output_size());
    }
    return func;
}

void runtime::gpu::GPU_Backend::initialize_io(void** target,
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

bool runtime::gpu::GPU_Backend::call(shared_ptr<Function> func,
                                     const vector<shared_ptr<runtime::Tensor>>& outputs,
                                     const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        throw runtime_error("compile() must be called before call().");
    }

    // ensure the GPURuntimeContext primitive pointers are valid
    m_context->prepare_runtime_context();

    // Device tensors
    initialize_io(instance.m_inputs.data(), inputs);
    initialize_io(instance.m_outputs.data(), outputs);

    auto ctx = m_context->m_runtime_context.get();
    instance.m_compiled_function(instance.m_inputs.data(), instance.m_outputs.data(), ctx);

    return true;
}

void runtime::gpu::GPU_Backend::remove_compiled_function(shared_ptr<Function> func)
{
    m_function_map.erase(func);
}

void runtime::gpu::GPU_Backend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function != nullptr)
    {
        throw runtime_error("Performance data collection must be enabled prior to compiling.");
    }
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::gpu::GPU_Backend::get_performance_data(shared_ptr<Function> func) const
{
    std::vector<runtime::PerformanceCounter> rc;
    auto it = m_function_map.find(func);
    if (it != m_function_map.end())
    {
        const FunctionInstance& instance = it->second;
        if (instance.m_external_function != nullptr)
        {
            auto* engine = instance.m_external_function->m_execution_engine.get();
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
    }
    return rc;
}

bool runtime::gpu::GPU_Backend::is_supported(const Node& node) const
{
    bool rc = true;

    // get op type
    element::Type type;
    if (node.description() == "Select")
    {
        type = node.get_input_element_type(1);
    }
    else if (node.description() == "Constant")
    {
        type = node.get_outputs().at(0).get_element_type();
    }
    else if (node.description() == "Parameter")
    {
        type = node.get_outputs().at(0).get_element_type();
    }
    else
    {
        type = node.get_input_element_type(0);
    }

    if (type != element::f32)
    {
        rc = false;
    }

    return rc;
}
