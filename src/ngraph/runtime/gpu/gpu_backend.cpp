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
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_executable.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_internal_function.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

extern "C" GPU_BACKEND_API void ngraph_register_gpu_backend()
{
    runtime::BackendManager::register_backend("GPU", [](const std::string& /* config */) {
        return make_shared<runtime::gpu::GPUBackend>();
    });
}

runtime::gpu::GPUBackend::GPUBackend()
    : runtime::Backend()
{
}

runtime::gpu::GPUBackend::BackendContext::BackendContext()
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

void runtime::gpu::GPUBackend::BackendContext::prepare_runtime_context()
{
    // set context current each time in case thread changed
    bind_cuda_context_to_thread();
    // add pointers to gpu primitives into the gpu runtime context
    m_runtime_context->gpu_primitives = m_primitive_emitter->get_primitives().data();
    m_runtime_context->gpu_memory_primitives = m_primitive_emitter->get_memory_primitives().data();
}

void runtime::gpu::GPUBackend::BackendContext::bind_cuda_context_to_thread()
{
    m_cuda_manager->SetContextCurrent();
}

runtime::gpu::GPUBackend::BackendContext::~BackendContext()
{
    cublasDestroy(*m_runtime_context->cublas_handle);
    delete m_runtime_context->cublas_handle;
    cudnnDestroy(*m_runtime_context->cudnn_handle);
    delete m_runtime_context->cudnn_handle;
    delete m_runtime_context->compiled_kernel_pool;
}

shared_ptr<runtime::Tensor>
    runtime::gpu::GPUBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::gpu::GPUTensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::gpu::GPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    if (memory_pointer != nullptr && !is_device_pointer(memory_pointer))
    {
        throw ngraph_error("The pointer passed to create_tensor is not a device pointer.");
    }
    return make_shared<runtime::gpu::GPUTensor>(element_type, shape, memory_pointer);
}

shared_ptr<runtime::Executable> runtime::gpu::GPUBackend::compile(shared_ptr<Function> func,
                                                                  bool timing_enable)
{
    shared_ptr<runtime::Executable> rc;
    auto it = m_exec_map.find(func);
    if (it != m_exec_map.end())
    {
        rc = it->second;
    }
    else
    {
        rc = make_shared<GPUExecutable>(func, timing_enable);
        m_exec_map.insert({func, rc});
    }
    return rc;
}

bool runtime::gpu::GPUBackend::is_supported(const Node& op) const
{
    set<string> unsupported_ops = {"Quantize",
                                   "Dequantize",
                                   "DynReplaceSlice",
                                   "DynReshape",
                                   "DynSlice",
                                   "ShapeOf",
                                   "All",
                                   "Any",
                                   "AllReduce",
                                   "BatchMatMul",
                                   "DynPad"
                                   "SelectAndScatter",
                                   "StopGradient",
                                   "EmbeddingLookup",
                                   "GenerateMask",
                                   "DynBroadcast",
                                   "Transpose",
                                   "Range",
                                   "Recv",
                                   "Send"};

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
