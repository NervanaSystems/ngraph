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

#pragma once

#include <plaidml/plaidml++.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/plaidml/plaidml_compilation_cache.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiler.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/plaidml/plaidml_executable.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            class PlaidML_Backend;
        }
    }
}

// Implements the runtime::Backend interface for the PlaidML nGraph backend.
class ngraph::runtime::plaidml::PlaidML_Backend final : public runtime::Backend
{
public:
    PlaidML_Backend(const std::string& configuration_string);
    ~PlaidML_Backend() final {}
    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) final;

    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(
        const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer) final;

    // N.B. The returned Executable will always be an instance of
    //      PlaidML_Executable, and may be safely converted via static
    //      casting.
    std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                        bool enable_performance_data = false) final;

    bool is_supported(const Node& node) const final;

    bool is_supported_property(const Property prop) const final;

    void remove_compiled_function(std::shared_ptr<Executable> exec) final;

private:
    Config m_config;
    Compiler m_compiler;
    CompilationCache m_cache;
};
