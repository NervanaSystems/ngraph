// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/runtime/gpu/manager.hpp"
#include "ngraph/runtime/gpu/backend.hpp"
#include "ngraph/runtime/gpu/external_function.hpp"

using namespace ngraph::runtime::gpu;

// std::shared_ptr<ngraph::runtime::Backend> GPUManager::allocate_backend()
// {
//     return std::make_shared<GPUBackend>();
// }

// // std::shared_ptr<ngraph::runtime::ExternalFunction>
// //     GPUManager::compile(const std::shared_ptr<ngraph::Function>& fun)
// // {
// //     return std::make_shared<GPUExternalFunction>(fun);
// // }

// ngraph::runtime::Manager::Factory GPUManager::factory = ngraph::runtime::Manager::register_factory(
//     "GPU", [](const std::string& name) -> std::shared_ptr<ngraph::runtime::Manager> {
//         return std::make_shared<GPUManager>();
//     });
