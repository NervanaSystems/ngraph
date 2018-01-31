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

#include <cstring>
#include <memory>

#include "cpu_tensor_view.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"

using namespace ngraph;
using namespace std;

runtime::cpu::CPUTensorView::CPUTensorView(const ngraph::element::Type& element_type,
                                           const Shape& shape,
                                           const string& name)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), name, true, true, false))
    , buffer(nullptr)
    , aligned_buffer(nullptr)
{
}

runtime::cpu::CPUTensorView::~CPUTensorView()
{
    free(buffer);
}

char* runtime::cpu::CPUTensorView::get_data_ptr()
{
    return aligned_buffer;
}

const char* runtime::cpu::CPUTensorView::get_data_ptr() const
{
    return aligned_buffer;
}

void runtime::cpu::CPUTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    if (tensor_offset + n > buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    char* target = get_data_ptr();
    memcpy(&target[tensor_offset], source, n);
}

void runtime::cpu::CPUTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    if (tensor_offset + n > buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }
    const char* source = get_data_ptr();
    memcpy(target, &source[tensor_offset], n);
}

size_t runtime::cpu::CPUTensorView::get_size() const
{
    return get_tensor_view_layout()->get_size();
}

const element::Type& runtime::cpu::CPUTensorView::get_element_type() const
{
    return get_tensor_view_layout()->get_element_type();
}
