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

#include <memory>

#include "ngraph/runtime/interpreter/cpu_backend.hpp"
#include "ngraph/runtime/interpreter/cpu_manager.hpp"
#include "ngraph/runtime/interpreter/external_function.hpp"

using namespace ngraph::runtime::interpreter;

std::shared_ptr<ngraph::runtime::Backend> CPUManager::allocate_backend()
{
    return std::make_shared<CPUBackend>();
}

std::shared_ptr<ngraph::runtime::ExternalFunction>
    CPUManager::compile(const std::shared_ptr<ngraph::Function>& fun)
{
    return std::make_shared<ExternalFunction>(fun);
}

ngraph::runtime::Manager::Factory CPUManager::factory = ngraph::runtime::Manager::register_factory(
    "CPU", [](const std::string& name) -> std::shared_ptr<ngraph::runtime::Manager> {
        return std::make_shared<CPUManager>();
    });
