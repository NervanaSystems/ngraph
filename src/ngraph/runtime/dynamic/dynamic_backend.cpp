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

#include "ngraph/runtime/dynamic/dynamic_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/shape_relevance.hpp"
#include "ngraph/specialize_shapes.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::dynamic::DynamicBackend::DynamicBackend(shared_ptr<runtime::Backend> wrapped_backend)
    : m_wrapped_backend(std::move(wrapped_backend))
{
}

shared_ptr<runtime::Tensor>
    runtime::dynamic::DynamicBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return m_wrapped_backend->create_tensor(type, shape);
}

shared_ptr<runtime::Tensor> runtime::dynamic::DynamicBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return m_wrapped_backend->create_tensor(type, shape, memory_pointer);
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic::DynamicBackend::create_dynamic_tensor(const element::Type& type,
                                                            const PartialShape& shape)
{
    return make_shared<DynamicTensor>(type, shape, m_wrapped_backend);
}

shared_ptr<runtime::Executable>
    runtime::dynamic::DynamicBackend::compile(shared_ptr<Function> function,
                                              bool enable_performance_collection)
{
    return make_shared<runtime::dynamic::DynamicExecutable>(
        function, m_wrapped_backend, enable_performance_collection);
}

runtime::dynamic::DynamicExecutable::DynamicExecutable(shared_ptr<Function> wrapped_function,
                                                       shared_ptr<runtime::Backend> wrapped_backend,
                                                       bool enable_performance_collection)
    : m_wrapped_function(wrapped_function)
    , m_wrapped_backend(wrapped_backend)
    , m_enable_performance_collection(enable_performance_collection)
{
    pass::Manager passes;
    passes.register_pass<pass::ShapeRelevance>();
    passes.run_passes(m_wrapped_function);

    set_parameters_and_results(*wrapped_function);
}

bool runtime::dynamic::DynamicExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    // TODO: Get cached executable out if it exists.
    // We will cache on:
    // (1) all shapes;
    // (2) all values of shape-relevant input tensors.

    std::vector<std::shared_ptr<runtime::Tensor>> wrapped_inputs;
    std::vector<element::Type> arg_element_types;
    std::vector<PartialShape> arg_shapes;

    for (auto& input : inputs)
    {
        if (auto dynamic_tensor = std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(input))
        {
            NGRAPH_CHECK(dynamic_tensor->has_storage());
            arg_element_types.push_back(dynamic_tensor->get_wrapped_tensor()->get_element_type());
            arg_shapes.push_back(dynamic_tensor->get_wrapped_tensor()->get_shape());
            wrapped_inputs.push_back(dynamic_tensor->get_wrapped_tensor());
        }
        else
        {
            arg_element_types.push_back(input->get_element_type());
            arg_shapes.push_back(input->get_shape());
            wrapped_inputs.push_back(input);
        }
    }

    // TODO: specialize_shapes needs to fill in values of shape-relevant params.
    auto clone = specialize_shapes(m_wrapped_function, arg_element_types, arg_shapes);
    // TODO: run constant folding and de-dynification on clone.
    const ResultVector& results = clone->get_results();
    NGRAPH_CHECK(results.size() == outputs.size());

    std::vector<std::shared_ptr<runtime::Tensor>> wrapped_outputs;

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (auto dynamic_tensor =
                std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(outputs[i]))
        {
            dynamic_tensor->make_storage(results[i]->get_output_element_type(0),
                                         results[i]->get_output_shape(0));
            wrapped_outputs.push_back(dynamic_tensor->get_wrapped_tensor());
        }
        else
        {
            wrapped_outputs.push_back(outputs[i]);
        }
    }

    // TODO: Put compiled executable in the cache.
    auto compiled_executable = m_wrapped_backend->compile(clone, m_enable_performance_collection);
    auto result = compiled_executable->call(wrapped_outputs, wrapped_inputs);

    return result;
}

runtime::dynamic::DynamicTensor::DynamicTensor(
    const element::Type& element_type,
    const PartialShape& shape,
    const std::shared_ptr<runtime::Backend>& wrapped_backend)
    : Tensor(make_shared<descriptor::Tensor>(element_type, shape, "wrapped_dynamic"))
    , m_wrapped_tensor(nullptr)
    , m_wrapped_backend(wrapped_backend)
{
}

const element::Type& runtime::dynamic::DynamicTensor::get_element_type() const
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

void runtime::dynamic::DynamicTensor::write(const void* p, size_t offset, size_t n)
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to write to a dynamic tensor with no allocated storage");
    m_wrapped_tensor->write(p, offset, n);
}

void runtime::dynamic::DynamicTensor::read(void* p, size_t offset, size_t n) const
{
    NGRAPH_CHECK(m_wrapped_tensor != nullptr,
                 "tried to read from a dynamic tensor with no allocated storage");
    m_wrapped_tensor->read(p, offset, n);
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

void runtime::dynamic::DynamicTensor::make_storage(const element::Type& element_type,
                                                   const Shape& shape)
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
