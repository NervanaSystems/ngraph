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

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

const Shape& runtime::Tensor::get_shape() const
{
    return m_descriptor->get_shape();
}

Strides runtime::Tensor::get_strides() const
{
    return m_descriptor->get_tensor_layout()->get_strides();
}

const element::Type& runtime::Tensor::get_element_type() const
{
    return m_descriptor->get_element_type();
}

shared_ptr<descriptor::layout::TensorLayout> runtime::Tensor::get_tensor_layout() const
{
    return m_descriptor->get_tensor_layout();
}

void runtime::Tensor::set_tensor_layout(const shared_ptr<descriptor::layout::TensorLayout>& layout)
{
    m_descriptor->set_tensor_layout(layout);
}

size_t runtime::Tensor::get_element_count() const
{
    return get_tensor_layout()->get_size();
}

size_t runtime::Tensor::get_size_in_bytes() const
{
    return get_tensor_layout()->get_size() * get_element_type().size();
}

const std::string& runtime::Tensor::get_name() const
{
    return m_descriptor->get_name();
}

bool runtime::Tensor::get_stale() const
{
    return m_stale;
}

void runtime::Tensor::set_stale(bool val)
{
    m_stale = val;
}

void runtime::Tensor::copy_to(runtime::Tensor* d, size_t offset, size_t n)
{
    void *copy = malloc(n);
    read(copy, offset, n); 
    d->write(copy, offset, n); 
}
