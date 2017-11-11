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

runtime::cpu::CPUTensorView::CPUTensorView(const ngraph::element::Type& element_type,
                                           const Shape& shape)
{
    size_t size = ngraph::shape_size(shape);
    size_t tensor_size = size * element_type.size();
    char* allocated;
    char* aligned;
    allocate_aligned_buffer(tensor_size, runtime::cpu::alignment, &allocated, &aligned);
    auto tmp = shared_ptr<char>(new char[size]);
    m_tensor_buffer = shared_ptr<char>(tmp, aligned);
}

void* runtime::cpu::CPUTensorView::get_data_ptr()
{
    return m_tensor_buffer.get();
}

void runtime::cpu::CPUTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    char* target = m_tensor_buffer.get();
    memcpy(&target[tensor_offset], source, n);
}

void runtime::cpu::CPUTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    const char* source = m_tensor_buffer.get();
    memcpy(target, &source[tensor_offset], n);
}
