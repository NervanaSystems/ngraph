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

#include "ngraph/runtime/dynamic/dynamic_tensor.hpp"

using namespace std;
using namespace ngraph;

runtime::dynamic::DynamicTensor::DynamicTensor(
    element::Type element_type,
    const PartialShape& shape,
    const std::shared_ptr<runtime::Backend>& wrapped_backend)
    : Tensor(make_shared<descriptor::Tensor>(element_type, shape, "wrapped_dynamic"))
    , m_wrapped_tensor(nullptr)
    , m_wrapped_backend(wrapped_backend)
{
}

Strides runtime::dynamic::DynamicTensor::get_strides() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "asked for strides of a dynamic tensor with no allocated storage");
    return ngraph::row_major_strides(m_wrapped_tensor->get_shape());
}

size_t runtime::dynamic::DynamicTensor::get_size_in_bytes() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "asked for size in bytes of a dynamic tensor with no allocated storage");
    return get_element_count() * get_element_type().size();
}

size_t runtime::dynamic::DynamicTensor::get_element_count() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "asked for element count of a dynamic tensor with no allocated storage");
    return shape_size(m_wrapped_tensor->get_shape());
}

element::Type runtime::dynamic::DynamicTensor::get_element_type() const
{
    if (m_wrapped_tensor == nullptr)
    {
        return m_descriptor->get_element_type();
    }
    else
    {
        return m_wrapped_tensor->get_element_type();
    }
}

const ngraph::Shape& runtime::dynamic::DynamicTensor::get_shape() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "asked for shape of a dynamic tensor with no allocated storage");
    return m_wrapped_tensor->get_shape();
}

void runtime::dynamic::DynamicTensor::write(const void* p, size_t n)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to write to a dynamic tensor with no allocated storage");
    m_wrapped_tensor->write(p, n);
}

void runtime::dynamic::DynamicTensor::read(void* p, size_t n) const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to read from a dynamic tensor with no allocated storage");
    m_wrapped_tensor->read(p, n);
}

void runtime::dynamic::DynamicTensor::copy_from(const ngraph::runtime::Tensor& source)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to copy_from to a dynamic tensor with no allocated storage");
    m_wrapped_tensor->copy_from(source);
}

bool runtime::dynamic::DynamicTensor::has_storage() const
{
    return m_wrapped_tensor != nullptr;
}

void runtime::dynamic::DynamicTensor::release_storage()
{
    m_wrapped_tensor = nullptr;
}

void runtime::dynamic::DynamicTensor::make_storage(element::Type element_type, const Shape& shape)
{
    NGRAPH_CHECK(element_type.is_static(), "make_storage requires a static element type");
    NGRAPH_CHECK(get_element_type().is_dynamic() || get_element_type() == element_type,
                 "tried to make storage with element type ",
                 element_type,
                 " which is incompatible with dynamic tensor element_type ",
                 get_element_type());
    NGRAPH_CHECK(get_partial_shape().relaxes(shape),
                 "tried to make storage with shape ",
                 shape,
                 " which is incompatible with dynamic tensor shape ",
                 get_partial_shape());
    m_wrapped_tensor = m_wrapped_backend->create_tensor(element_type, shape);
}

const std::shared_ptr<ngraph::runtime::Tensor>&
    runtime::dynamic::DynamicTensor::get_wrapped_tensor() const
{
    return m_wrapped_tensor;
}
