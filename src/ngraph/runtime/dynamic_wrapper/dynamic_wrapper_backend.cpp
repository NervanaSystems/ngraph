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

#include "ngraph/runtime/dynamic_wrapper/dynamic_wrapper_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::dynamic_wrapper::DynamicWrapperBackend::DynamicWrapperBackend(
    unique_ptr<runtime::Backend> wrapped_backend)
    : m_wrapped_backend(std::move(wrapped_backend))
{
}

shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(const element::Type& type,
                                                                   const Shape& shape)
{
    NGRAPH_CHECK(type.is_static());
    return m_wrapped_backend->create_tensor(type, shape);
}

shared_ptr<runtime::Tensor> runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    NGRAPH_CHECK(type.is_static());
    return m_wrapped_backend->create_tensor(type, shape, memory_pointer);
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_dynamic_tensor(
        const element::Type& type, const PartialShape& shape)
{
    if (type.is_static() && shape.is_static())
    {
        return m_wrapped_backend->create_tensor(type, shape.to_shape());
    }
    else
    {
        return make_shared<DynamicTensor>(type, shape, this, m_wrapped_backend);
    }
}

shared_ptr<runtime::Executable>
    runtime::dynamic_wrapper::DynamicWrapperBackend::compile(shared_ptr<Function> function,
                                                             bool enable_performance_collection)
{
    // TODO(amprocte): don't want to just delegate this. Need to produce a wrapped executable.
    return m_wrapped_backend->compile(function, enable_performance_collection);
}

runtime::dynamic_wrapper::DynamicTensor::DynamicTensor(
    const element::Type& element_type,
    const PartialShape& shape,
    const runtime::Backend* parent,
    const std::shared_ptr<ngraph::runtime::Backend>& wrapped_backend)
    : Tensor(make_shared<descriptor::Tensor>(element_type, shape, "dynamic_external"), parent)
    , m_wrapped_tensor(nullptr)
    , m_wrapped_backend(wrapped_backend)
{
}

const ngraph::Shape& runtime::dynamic_wrapper::DynamicTensor::get_shape() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr);
    return m_wrapped_tensor->get_shape();
}

void runtime::dynamic_wrapper::DynamicTensor::write(const void* p, size_t offset, size_t n)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr);
    m_wrapped_tensor->write(p, offset, n);
}

void runtime::dynamic_wrapper::DynamicTensor::read(void* p, size_t offset, size_t n) const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr);
    m_wrapped_tensor->read(p, offset, n);
}

void runtime::dynamic_wrapper::DynamicTensor::copy_from(const ngraph::runtime::Tensor& source)
{
    NGRAPH_CHECK(source.get_element_type().is_static() && source.get_partial_shape().is_static());
    clear_type_and_shape();
    set_type_and_shape(source.get_element_type(), source.get_shape());
    m_wrapped_tensor->copy_from(source);
}

void runtime::dynamic_wrapper::DynamicTensor::set_type_and_shape(const element::Type& et,
                                                                 const Shape& shape)
{
    NGRAPH_CHECK(et.is_static());
    NGRAPH_CHECK(get_partial_shape().is_dynamic() || get_element_type().is_dynamic());
    NGRAPH_CHECK(m_wrapped_tensor == nullptr);
    m_wrapped_tensor = m_wrapped_backend->create_tensor(et, shape);
}

void runtime::dynamic_wrapper::DynamicTensor::clear_type_and_shape()
{
    NGRAPH_CHECK(get_partial_shape().is_dynamic() || get_element_type().is_dynamic());
    m_wrapped_tensor = nullptr;
}
