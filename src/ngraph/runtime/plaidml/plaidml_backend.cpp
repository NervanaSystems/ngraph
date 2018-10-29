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

#include "ngraph/runtime/plaidml/plaidml_backend.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiled_function.hpp"
#include "ngraph/runtime/plaidml/plaidml_tensor.hpp"
#include "ngraph/util.hpp"

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::PlaidML_Backend::PlaidML_Backend(const char* configuration_string)
    : m_config{parse_config_string(configuration_string)}
    , m_compiler{&m_config}
{
}

std::shared_ptr<ngraph::runtime::Tensor> ngraph::runtime::plaidml::PlaidML_Backend::create_tensor(
    const ngraph::element::Type& element_type, const ngraph::Shape& shape)
{
    return std::make_shared<PlaidML_Tensor>(&m_config, element_type, shape, "direct_data", nullptr);
}

std::shared_ptr<ngraph::runtime::Tensor> ngraph::runtime::plaidml::PlaidML_Backend::create_tensor(
    const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return std::make_shared<PlaidML_Tensor>(
        &m_config, element_type, shape, "direct_data", memory_pointer);
}

bool ngraph::runtime::plaidml::PlaidML_Backend::compile(std::shared_ptr<Function> func)
{
    m_cache.compile(func, &m_compiler);
    return true;
}

bool ngraph::runtime::plaidml::PlaidML_Backend::call(
    std::shared_ptr<Function> func,
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    auto cfunc = m_cache.try_lookup(func);
    if (!cfunc)
    {
        cfunc = m_compiler.compile(func);
    }
    cfunc->schedule_invocation(inputs, outputs);
    return true;
}

void ngraph::runtime::plaidml::PlaidML_Backend::remove_compiled_function(
    std::shared_ptr<Function> func)
{
    m_cache.forget(func);
}

void ngraph::runtime::plaidml::PlaidML_Backend::save(std::shared_ptr<Function> func,
                                                     const std::string& filename,
                                                     plaidml_file_format format)
{
    auto cfunc = m_cache.try_lookup(func);
    if (!cfunc)
    {
        cfunc = m_compiler.compile(func);
    }
    cfunc->save(filename, format);
}

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" ngraph::runtime::Backend* new_backend(const char* configuration_string)
{
    return new ngraph::runtime::plaidml::PlaidML_Backend{configuration_string};
}

extern "C" void delete_backend(ngraph::runtime::Backend* backend)
{
    delete backend;
}
