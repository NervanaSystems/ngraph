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
    return make_shared<WrappedStaticTensor>(m_wrapped_backend->create_tensor(type, shape), this);
}

shared_ptr<runtime::Tensor> runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<WrappedStaticTensor>(
        m_wrapped_backend->create_tensor(type, shape, memory_pointer), this);
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_dynamic_tensor(
        const element::Type& type, const PartialShape& shape)
{
    throw std::invalid_argument("create_dynamic_tensor broken for now");
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
    set_parameters_and_results(*wrapped_function);
}

bool runtime::dynamic_wrapper::WrappedExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    std::vector<std::shared_ptr<runtime::Tensor>> real_outputs;
    std::vector<std::shared_ptr<runtime::Tensor>> real_inputs;

    for (auto& output : outputs)
    {
        real_outputs.push_back(
            std::static_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(output)
                ->get_wrapped_tensor());
    }
    for (auto& input : inputs)
    {
        real_inputs.push_back(
            std::static_pointer_cast<runtime::dynamic_wrapper::WrappedStaticTensor>(input)
                ->get_wrapped_tensor());
    }

    // For now we cache nothing!
    auto compiled_executable = m_wrapped_backend->compile(m_wrapped_function);
    auto result = compiled_executable->call(real_outputs, real_inputs);

    return result;
}

runtime::dynamic_wrapper::WrappedStaticTensor::WrappedStaticTensor(
    const std::shared_ptr<runtime::Tensor>& wrapped_tensor, const runtime::Backend* parent)
    : Tensor(make_shared<descriptor::Tensor>(
                 wrapped_tensor->get_element_type(), wrapped_tensor->get_shape(), "wrapped_static"),
             parent)
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
