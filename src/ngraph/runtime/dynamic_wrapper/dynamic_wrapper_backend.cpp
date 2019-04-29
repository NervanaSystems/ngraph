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
    return make_shared<WrappedStaticTensor>(m_wrapped_backend->create_tensor(type, shape));
}

shared_ptr<runtime::Tensor> runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<WrappedStaticTensor>(
        m_wrapped_backend->create_tensor(type, shape, memory_pointer));
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_dynamic_tensor(
        const element::Type& type, const PartialShape& shape)
{
    return make_shared<WrappedDynamicTensor>(type, shape, m_wrapped_backend);
}

shared_ptr<runtime::Executable>
    runtime::dynamic_wrapper::DynamicWrapperBackend::compile(shared_ptr<Function> function,
                                                             bool enable_performance_collection)
{
    return make_shared<runtime::dynamic_wrapper::WrappedExecutable>(
        function, m_wrapped_backend, enable_performance_collection);
}

runtime::dynamic_wrapper::WrappedExecutable::WrappedExecutable(
    shared_ptr<Function> wrapped_function,
    shared_ptr<runtime::Backend> wrapped_backend,
    bool enable_performance_collection)
    : m_wrapped_function(wrapped_function)
    , m_wrapped_backend(wrapped_backend)
    , m_enable_performance_collection(enable_performance_collection)
{
    // TODO: Run relevance analysis here.
    set_parameters_and_results(*wrapped_function);
}

bool runtime::dynamic_wrapper::WrappedExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    // TODO: Get cached executable out if it exists.

    // TODO: Run shape inference passes here.
    // TODO: Put executable in the cache.
    auto compiled_executable = m_wrapped_backend->compile(m_wrapped_function);

    std::vector<std::shared_ptr<runtime::Tensor>> real_outputs;
    std::vector<std::shared_ptr<runtime::Tensor>> real_inputs;

    for (auto& output : outputs)
    {
        // TODO: If dynamic and no storage of suitable shape, make storage
        NGRAPH_CHECK(
            std::dynamic_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(output));

        real_outputs.push_back(
            std::static_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(output)
                ->get_wrapped_tensor());
    }
    for (auto& input : inputs)
    {
        // TODO: If dynamic and no storage, bail
        NGRAPH_CHECK(
            std::dynamic_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(input));

        real_inputs.push_back(
            std::static_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(input)
                ->get_wrapped_tensor());
    }

    auto result = compiled_executable->call(real_outputs, real_inputs);

    return result;
}

runtime::dynamic_wrapper::WrappedStaticTensor::WrappedStaticTensor(
    const std::shared_ptr<runtime::Tensor>& wrapped_tensor)
    : Tensor(make_shared<descriptor::Tensor>(
          wrapped_tensor->get_element_type(), wrapped_tensor->get_shape(), "wrapped_static"))
    , m_wrapped_tensor(wrapped_tensor)
{
}

const ngraph::Shape& runtime::dynamic_wrapper::WrappedStaticTensor::get_shape() const
{
    return m_wrapped_tensor->get_shape();
}

void runtime::dynamic_wrapper::WrappedStaticTensor::write(const void* p, size_t offset, size_t n)
{
    m_wrapped_tensor->write(p, offset, n);
}

void runtime::dynamic_wrapper::WrappedStaticTensor::read(void* p, size_t offset, size_t n) const
{
    m_wrapped_tensor->read(p, offset, n);
}

void runtime::dynamic_wrapper::WrappedStaticTensor::copy_from(const ngraph::runtime::Tensor& source)
{
    m_wrapped_tensor->copy_from(source);
}

const std::shared_ptr<ngraph::runtime::Tensor>&
    runtime::dynamic_wrapper::WrappedStaticTensor::get_wrapped_tensor() const
{
    return m_wrapped_tensor;
}

runtime::dynamic_wrapper::WrappedDynamicTensor::WrappedDynamicTensor(
    const element::Type& element_type,
    const PartialShape& shape,
    const std::shared_ptr<runtime::Backend>& wrapped_backend)
    : Tensor(make_shared<descriptor::Tensor>(element_type, shape, "wrapped_dynamic"))
    , m_wrapped_tensor(nullptr)
    , m_wrapped_backend(wrapped_backend)
{
}

const ngraph::Shape& runtime::dynamic_wrapper::WrappedDynamicTensor::get_shape() const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "asked for shape of a dynamic tensor with no allocated storage");
    return m_wrapped_tensor->get_shape();
}

void runtime::dynamic_wrapper::WrappedDynamicTensor::write(const void* p, size_t offset, size_t n)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to write to a dynamic tensor with no allocated storage");
    m_wrapped_tensor->write(p, offset, n);
}

void runtime::dynamic_wrapper::WrappedDynamicTensor::read(void* p, size_t offset, size_t n) const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to read from a dynamic tensor with no allocated storage");
    m_wrapped_tensor->read(p, offset, n);
}

void runtime::dynamic_wrapper::WrappedDynamicTensor::copy_from(
    const ngraph::runtime::Tensor& source)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to copy_from to a dynamic tensor with no allocated storage");
    m_wrapped_tensor->copy_from(source);
}

bool runtime::dynamic_wrapper::WrappedDynamicTensor::has_storage() const
{
    return m_wrapped_tensor != nullptr;
}

void runtime::dynamic_wrapper::WrappedDynamicTensor::release_storage()
{
    m_wrapped_tensor = nullptr;
}

void runtime::dynamic_wrapper::WrappedDynamicTensor::make_storage(const element::Type& element_type,
                                                                  const Shape& shape)
{
    NGRAPH_CHECK(element_type.is_static(), "make_storage requires a static element type");
    m_wrapped_tensor = m_wrapped_backend->create_tensor(element_type, shape);
}

const std::shared_ptr<ngraph::runtime::Tensor>&
    runtime::dynamic_wrapper::WrappedDynamicTensor::get_wrapped_tensor() const
{
    return m_wrapped_tensor;
}
