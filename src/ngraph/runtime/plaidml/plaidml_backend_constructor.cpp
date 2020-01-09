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

#include "ngraph/runtime/plaidml/plaidml_backend.hpp"

#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/plaidml/plaidml_backend_visibility.hpp"

extern "C" PLAIDML_BACKEND_API void ngraph_register_plaidml_backend()
{
    ngraph::runtime::BackendManager::register_backend("PlaidML", [](const std::string& config) {
        return std::make_shared<ngraph::runtime::plaidml::PlaidML_Backend>(config);
    });
}
