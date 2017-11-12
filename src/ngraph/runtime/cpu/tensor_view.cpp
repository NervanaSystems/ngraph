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

#include "cpu_backend.hpp"
#include "tensor_view.hpp"

using namespace ngraph;
using namespace std;

extern "C" void
    allocate_aligned_buffer(size_t size, size_t alignment, char** allocated, char** aligned_ptr);
extern "C" void free_aligned_buffer(void* allocated);

runtime::cpu::CPUTensorView::CPUTensorView(const ngraph::element::Type& element_type,
                                           const Shape& shape)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), "external", true, true))
{
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size() * element_type.size();
    allocate_aligned_buffer(m_buffer_size, runtime::cpu::alignment, &m_allocated, &m_buffer);
}

runtime::cpu::CPUTensorView::~CPUTensorView()
{
    free_aligned_buffer(m_allocated);
}

char* runtime::cpu::CPUTensorView::get_data_ptr()
{
    return m_buffer;
}

const char* runtime::cpu::CPUTensorView::get_data_ptr() const
{
    return m_buffer;
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
