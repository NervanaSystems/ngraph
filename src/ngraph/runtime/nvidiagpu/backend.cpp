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
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/nvidiagpu/backend.hpp"
#include "ngraph/runtime/nvidiagpu/external_function.hpp"
#include "ngraph/runtime/nvidiagpu/internal_function.hpp"
#include "ngraph/runtime/nvidiagpu/primitive_emitter.hpp"
#include "ngraph/runtime/nvidiagpu/tensor.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::nvidiagpu::Backend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::nvidiagpu::Backend::Backend()
    : runtime::Backend()
    , m_context(new BackendContext())
{
}

runtime::nvidiagpu::Backend::BackendContext::BackendContext()
    : m_runtime_context(new RuntimeContext)
    , m_primitive_emitter(new PrimitiveEmitter(m_runtime_context))
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

void runtime::nvidiagpu::Backend::BackendContext::prepare_runtime_context()
{
    // set context current each time in case thread changed
    bind_cuda_context_to_thread();
    // add pointers to nvidiagpu primitives into the nvidiagpu runtime context
    m_runtime_context->nvidiagpu_primitives = m_primitive_emitter->get_primitives().data();
    m_runtime_context->nvidiagpu_memory_primitives =
        m_primitive_emitter->get_memory_primitives().data();
}

void runtime::nvidiagpu::Backend::BackendContext::bind_cuda_context_to_thread()
{
    m_cuda_manager->SetContextCurrent();
}

runtime::nvidiagpu::Backend::BackendContext::~BackendContext()
{
    cublasDestroy(*m_runtime_context->cublas_handle);
    delete m_runtime_context->cublas_handle;
    cudnnDestroy(*m_runtime_context->cudnn_handle);
    delete m_runtime_context->cudnn_handle;
    delete m_runtime_context->compiled_kernel_pool;
}

shared_ptr<runtime::Tensor>
    runtime::nvidiagpu::Backend::create_tensor(const element::Type& element_type,
                                               const ngraph::Shape& shape)
{
    return make_shared<runtime::nvidiagpu::Tensor>(element_type, shape, this);
}

shared_ptr<runtime::Tensor> runtime::nvidiagpu::Backend::create_tensor(
    const element::Type& element_type, const ngraph::Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::nvidiagpu::Tensor>(element_type, shape, memory_pointer, this);
}

runtime::Handle runtime::nvidiagpu::Backend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_compiled_function == nullptr)
    {
        m_context->bind_cuda_context_to_thread();
        instance.m_compiled_function = runtime::nvidiagpu::CompiledFunction::make(func, m_context);
        instance.m_compiled_function->m_emit_timing = instance.m_performance_counters_enabled;
        instance.m_compiled_function->compile();
        instance.m_runtime = instance.m_compiled_function->m_runtime;
        instance.m_inputs.resize(func->get_parameters().size());
        instance.m_outputs.resize(func->get_output_size());
    }
    return func;
}

void runtime::nvidiagpu::Backend::initialize_io(void** target,
                                                const vector<shared_ptr<runtime::Tensor>>& source)
{
    for (size_t i = 0; i < source.size(); i++)
    {
        shared_ptr<runtime::nvidiagpu::Tensor> tv =
            dynamic_pointer_cast<runtime::nvidiagpu::Tensor>(source[i]);
        if (tv)
        {
            target[i] = tv->m_allocated_buffer_pool;
        }
        else
        {
            throw invalid_argument("Tensors passed to NVIDIAGPU backend must be NVIDIAGPU Tensors");
        }
    }
}

bool runtime::nvidiagpu::Backend::call(shared_ptr<Function> func,
                                       const vector<shared_ptr<runtime::Tensor>>& outputs,
                                       const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_compiled_function == nullptr)
    {
        throw runtime_error("compile() must be called before call().");
    }

    // ensure the RuntimeContext primitive pointers are valid
    m_context->prepare_runtime_context();

    // Device tensors
    initialize_io(instance.m_inputs.data(), inputs);
    initialize_io(instance.m_outputs.data(), outputs);

    auto ctx = m_context->m_runtime_context.get();
    instance.m_runtime(instance.m_inputs.data(), instance.m_outputs.data(), ctx);

    return true;
}

void runtime::nvidiagpu::Backend::remove_compiled_function(shared_ptr<Function> func)
{
    m_function_map.erase(func);
}

void runtime::nvidiagpu::Backend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_compiled_function != nullptr)
    {
        throw runtime_error("Performance data collection must be enabled prior to compiling.");
    }
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::nvidiagpu::Backend::get_performance_data(shared_ptr<Function> func) const
{
    std::vector<runtime::PerformanceCounter> rc;
    auto it = m_function_map.find(func);
    if (it != m_function_map.end())
    {
        const FunctionInstance& instance = it->second;
        if (instance.m_compiled_function != nullptr)
        {
            instance.m_compiled_function->get_performance_data(rc);
        }
    }
    return rc;
}

bool runtime::nvidiagpu::Backend::is_supported(const Node& op) const
{
    set<string> unsupported_ops = {"Quantize",
                                   "Dequantize",
                                   "ngraph::ShapeOf",
                                   "All",
                                   "Any",
                                   "AllReduce",
                                   "SelectAndScatter",
                                   "StopGradient",
                                   "EmbeddingLookup",
                                   "GenerateMask"};

    set<string> float_only = {"MaxPoolBackprop", "AvgPoolBackprop", "MaxPool", "Dot"};

    if (unsupported_ops.find(op.description()) != unsupported_ops.end())
    {
        return false;
    }

    if (float_only.find(op.description()) != float_only.end())
    {
        if (op.get_output_element_type(0) != element::f32 &&
            op.get_output_element_type(0) != element::f64)
        {
            return false;
        }
    }

    if (op.description() == "BatchNormInference")
    {
        const ngraph::op::BatchNormInference* bn =
            static_cast<const ngraph::op::BatchNormInference*>(&op);
        if (bn->get_eps_value() < CUDNN_BN_MIN_EPSILON)
        {
            return false;
        }
    }
    else if (op.description() == "BatchNormTraining")
    {
        const ngraph::op::BatchNormTraining* bn =
            static_cast<const ngraph::op::BatchNormTraining*>(&op);
        if (bn->get_eps_value() < CUDNN_BN_MIN_EPSILON)
        {
            return false;
        }
    }

    return true;
}
