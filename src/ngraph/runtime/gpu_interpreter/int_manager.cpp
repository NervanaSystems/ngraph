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

#include "ngraph/runtime/gpu_interpreter/int_backend.hpp"
#include "ngraph/runtime/gpu_interpreter/int_external_function.hpp"
#include "ngraph/runtime/gpu_interpreter/int_manager.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::Backend> runtime::gpu_interpreter::INT_Manager::allocate_backend()
{
    return make_shared<INT_Backend>();
}

shared_ptr<runtime::ExternalFunction>
    runtime::gpu_interpreter::INT_Manager::compile(const shared_ptr<Function>& fun)
{
    return make_shared<ExternalFunction>(fun);
}

runtime::Manager::Factory runtime::gpu_interpreter::INT_Manager::factory =
    runtime::Manager::register_factory("gpu_interpreter",
                                       [](const string& name) -> shared_ptr<runtime::Manager> {
                                           return make_shared<INT_Manager>();
                                       });
