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

#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"
#include "ngraph/runtime/ngvm/ngvm_transformer.hpp"

using namespace ngraph::runtime::ngvm;

std::shared_ptr<ngraph::runtime::Backend> NGVMTransformer::allocate_backend()
{
    return std::make_shared<NGVMBackend>();
}

std::shared_ptr<ngraph::runtime::ExternalFunction>
    NGVMTransformer::compile(const std::shared_ptr<ngraph::Function>& fun)
{
    return std::make_shared<ExternalFunction>(fun);
}

ngraph::runtime::Transformer::Factory NGVMTransformer::factory =
    ngraph::runtime::Transformer::register_factory(
        "NGVM", [](const std::string& name) -> std::shared_ptr<ngraph::runtime::Transformer> {
            return std::make_shared<NGVMTransformer>();
        });
