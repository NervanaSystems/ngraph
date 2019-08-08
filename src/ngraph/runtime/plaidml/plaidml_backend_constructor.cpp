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

#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/plaidml/plaidml_backend.hpp"
#include "ngraph/runtime/plaidml/plaidml_backend_visibility.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            class PlaidML_BackendConstructor;
        }
    }
}

class ngraph::runtime::plaidml::PlaidML_BackendConstructor final
    : public runtime::BackendConstructor
{
public:
    ~PlaidML_BackendConstructor() final {}
    std::shared_ptr<Backend> create(const std::string& config) final;
};

std::shared_ptr<ngraph::runtime::Backend>
    ngraph::runtime::plaidml::PlaidML_BackendConstructor::create(const std::string& config)
{
    return std::make_shared<PlaidML_Backend>(config);
}

extern "C" PLAIDML_BACKEND_API ngraph::runtime::BackendConstructor*
    get_backend_constructor_pointer()
{
    static ngraph::runtime::plaidml::PlaidML_BackendConstructor backend_constructor;
    return &backend_constructor;
}
