// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/tensor_view.hpp"

using namespace ngraph;
using namespace std;

runtime::cpu::CPUTensorView::CPUTensorView(const ngraph::element::Type& element_type,
                                           const Shape& shape)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), "external", true, true))
    , m_allocated_buffer_pool(nullptr)
    , m_aligned_buffer_pool(nullptr)

{
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size() * element_type.size();
    if (m_buffer_size > 0)
    {
        size_t allocation_size = m_buffer_size + runtime::cpu::alignment;
        m_allocated_buffer_pool = static_cast<char*>(malloc(allocation_size));
        m_aligned_buffer_pool = m_allocated_buffer_pool;
        size_t mod = size_t(m_aligned_buffer_pool) % alignment;
        if (mod != 0)
        {
            m_aligned_buffer_pool += (alignment - mod);
        }
    }
}

runtime::cpu::CPUTensorView::~CPUTensorView()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        free(m_allocated_buffer_pool);
    }
}

char* runtime::cpu::CPUTensorView::get_data_ptr()
{
    return m_aligned_buffer_pool;
}

const char* runtime::cpu::CPUTensorView::get_data_ptr() const
{
    return m_aligned_buffer_pool;
}

void runtime::cpu::CPUTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    if (tensor_offset + n > m_buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    char* target = get_data_ptr();
    memcpy(&target[tensor_offset], source, n);
}

void runtime::cpu::CPUTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    if (tensor_offset + n > m_buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }
    const char* source = get_data_ptr();
    memcpy(target, &source[tensor_offset], n);
}
