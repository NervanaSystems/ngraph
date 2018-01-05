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

#include "ngraph/runtime/gpu/gpu_manager.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"

using namespace ngraph::runtime::gpu;

std::shared_ptr<ngraph::runtime::Backend> GPU_Manager::allocate_backend()
{
    return std::make_shared<GPU_Backend>();
}

std::shared_ptr<ngraph::runtime::ExternalFunction>
    GPU_Manager::compile(const std::shared_ptr<ngraph::Function>& fun)
{
    return std::make_shared<GPU_ExternalFunction>(fun);
}

ngraph::runtime::Manager::Factory GPU_Manager::factory = ngraph::runtime::Manager::register_factory(
    "GPU", [](const std::string& name) -> std::shared_ptr<ngraph::runtime::Manager> {
        return std::make_shared<GPU_Manager>();
    });
