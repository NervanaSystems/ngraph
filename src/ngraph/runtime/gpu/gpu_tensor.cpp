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

#include <memory>

#include <cuda_runtime.h>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/gpu/cuda_error_check.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::GPUTensor::GPUTensor(const ngraph::element::Type& element_type,
                                   const Shape& shape,
                                   void* memory_pointer)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, ""))
    , m_custom_memory(false)
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));

    m_buffer_size = shape_size(shape) * element_type.size();
    if (memory_pointer != nullptr)
    {
        if (is_device_pointer(memory_pointer))
        {
            m_allocated_buffer_pool = memory_pointer;
            m_custom_memory = true;
        }
        else
        {
            throw ngraph_error("The pointer passed to GPUTensor is not a device pointer.");
        }
    }
    else if (m_buffer_size > 0)
    {
        m_allocated_buffer_pool = runtime::gpu::create_gpu_buffer(m_buffer_size);
    }
}

runtime::gpu::GPUTensor::GPUTensor(const ngraph::element::Type& element_type, const Shape& shape)
    : GPUTensor(element_type, shape, nullptr)
{
}

runtime::gpu::GPUTensor::~GPUTensor()
{
    if (!m_custom_memory && (m_allocated_buffer_pool != nullptr))
    {
        runtime::gpu::free_gpu_buffer(m_allocated_buffer_pool);
    }
}

void runtime::gpu::GPUTensor::write(const void* source, size_t n_bytes)
{
    runtime::gpu::cuda_memcpyHtD(m_allocated_buffer_pool, source, n_bytes);
}

void runtime::gpu::GPUTensor::read(void* target, size_t n_bytes) const
{
    runtime::gpu::cuda_memcpyDtH(target, m_allocated_buffer_pool, n_bytes);
}

void runtime::gpu::GPUTensor::copy_from(const runtime::Tensor& source)
{
    try
    {
        const GPUTensor& src = dynamic_cast<const GPUTensor&>(source);

        if (get_element_count() != src.get_element_count())
        {
            throw invalid_argument("runtime::gpu::GPUTensor::copy_from element count must match.");
        }
        if (get_element_type() != src.get_element_type())
        {
            throw invalid_argument("runtime::gpu::GPUTensor::copy_from element types must match.");
        }
        runtime::gpu::cuda_memcpyDtD(
            m_allocated_buffer_pool, src.m_allocated_buffer_pool, source.get_size_in_bytes());
    }
    catch (const std::bad_cast& e)
    {
        throw invalid_argument(
            "runtime::gpu::GPUTensor::copy_from source must be a GPUTensor. ErrMsg:\n" +
            std::string(e.what()));
    }
}
