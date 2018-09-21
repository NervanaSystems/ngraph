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

#include <memory>

#include <cuda_runtime.h>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::GPU_TensorView::GPU_TensorView(const ngraph::element::Type& element_type,
                                             const Shape& shape,
                                             void* memory_pointer)
    : runtime::TensorView(
          std::make_shared<ngraph::descriptor::TensorView>(element_type, shape, "external"))
    , m_custom_memory(false)
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));

    m_buffer_size = shape_size(shape) * element_type.size();
    if (memory_pointer != nullptr)
    {
        m_allocated_buffer_pool = memory_pointer;
        m_custom_memory = true;
    }
    else if (m_buffer_size > 0)
    {
        CUDA_RT_SAFE_CALL(cudaMalloc(static_cast<void**>(&m_allocated_buffer_pool), m_buffer_size));
    }
}

runtime::gpu::GPU_TensorView::GPU_TensorView(const ngraph::element::Type& element_type,
                                             const Shape& shape)
    : GPU_TensorView(element_type, shape, nullptr)
{
}

runtime::gpu::GPU_TensorView::~GPU_TensorView()
{
    if (!m_custom_memory && (m_allocated_buffer_pool != nullptr))
    {
        CUDA_RT_SAFE_CALL_NO_THROW(cudaFree(m_allocated_buffer_pool));
    }
}

void runtime::gpu::GPU_TensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(m_allocated_buffer_pool, source, n, cudaMemcpyHostToDevice));
}

void runtime::gpu::GPU_TensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(target, m_allocated_buffer_pool, n, cudaMemcpyDeviceToHost));
}
