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

#include "ngraph/runtime/dynamic/dynamic_backend.hpp"
#include "ngraph/runtime/dynamic/dynamic_executable.hpp"
#include "ngraph/runtime/dynamic/dynamic_tensor.hpp"

using namespace std;
using namespace ngraph;

runtime::dynamic::DynamicBackend::DynamicBackend(shared_ptr<runtime::Backend> wrapped_backend)
    : m_wrapped_backend(std::move(wrapped_backend))
{
}

shared_ptr<runtime::Tensor> runtime::dynamic::DynamicBackend::create_tensor()
{
    return m_wrapped_backend->create_tensor();
}

shared_ptr<runtime::Tensor> runtime::dynamic::DynamicBackend::create_tensor(element::Type type,
                                                                            const Shape& shape)
{
    return m_wrapped_backend->create_tensor(type, shape);
}

shared_ptr<runtime::Tensor> runtime::dynamic::DynamicBackend::create_tensor(element::Type type,
                                                                            const Shape& shape,
                                                                            void* memory_pointer)
{
    return m_wrapped_backend->create_tensor(type, shape, memory_pointer);
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic::DynamicBackend::create_dynamic_tensor(element::Type type,
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
