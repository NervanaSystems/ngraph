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

using descriptor::layout::DenseTensorLayout;

// To silence warnings in vscode.
#if !defined(NGRAPH_VERSION)
#define NGRAPH_VERSION ""
#endif

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    std::unique_ptr<runtime::Backend> wrapped_backend = nullptr;

    // Skip past the colon in the configuration string (or stop at the end if there isn't one).
    while (*configuration_string != '\0' && *configuration_string != ':')
    {
        configuration_string++;
    }

    if (*configuration_string == ':')
    {
        configuration_string++;
    }

    // Figure out what wrapped backend to use.

    // If something is specified in the configuration string, we will always try to use that and
    // fail if we can't.
    if (*configuration_string != '\0')
    {
        wrapped_backend = runtime::Backend::create(configuration_string);
    }
    // Otherwise we will check for the NGRAPH_DYNAMIC_WRAPPER_WRAPPED_BACKEND environment
    // variable. If it is set, we will try to use that value; otherwise, we will defualt to
    // INTERPRETER.
    else
    {
        auto env_var_requested_backend = std::getenv("NGRAPH_DYNAMIC_WRAPPER_WRAPPED_BACKEND");

        if (env_var_requested_backend != nullptr)
        {
            wrapped_backend = runtime::Backend::create(env_var_requested_backend);
        }
        else
        {
            wrapped_backend = runtime::Backend::create("INTERPRETER");
        }
    }

    // If we were unable to create the wrapped backend, created of the wrapper backend fails.
    if (wrapped_backend == nullptr)
    {
        return nullptr;
    }
    else
    {
        return new runtime::dynamic_wrapper::DynamicWrapperBackend(std::move(wrapped_backend));
    }
}

runtime::dynamic_wrapper::DynamicWrapperBackend::DynamicWrapperBackend(
    unique_ptr<runtime::Backend> wrapped_backend)
    : m_wrapped_backend(std::move(wrapped_backend))
{
}

shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(const element::Type& type,
                                                                   const Shape& shape)
{
    return m_wrapped_backend->create_tensor(type, shape);
}

shared_ptr<runtime::Tensor> runtime::dynamic_wrapper::DynamicWrapperBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return m_wrapped_backend->create_tensor(type, shape, memory_pointer);
}

std::shared_ptr<runtime::Tensor>
    runtime::dynamic_wrapper::DynamicWrapperBackend::create_dynamic_tensor(
        const element::Type& type, const PartialShape& shape)
{
    return m_wrapped_backend->create_dynamic_tensor(type, shape);
}

shared_ptr<runtime::Executable>
    runtime::dynamic_wrapper::DynamicWrapperBackend::compile(shared_ptr<Function> function,
                                                             bool enable_performance_collection)
{
    return m_wrapped_backend->compile(function, enable_performance_collection);
}
