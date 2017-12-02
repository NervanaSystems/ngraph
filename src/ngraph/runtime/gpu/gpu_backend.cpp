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

#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/external_function.hpp"

using namespace ngraph::runtime::gpu;

std::shared_ptr<ngraph::runtime::CallFrame>
    GPUBackend::make_call_frame(const std::shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}
