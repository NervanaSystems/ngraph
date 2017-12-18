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
#include <string>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/runtime/gpu/call_frame.hpp"
#include "ngraph/runtime/gpu/external_function.hpp"

using namespace std;
using namespace ngraph::runtime::gpu;
using namespace ngraph;

ngraph::runtime::gpu::ExternalFunction::ExternalFunction(
    const std::shared_ptr<ngraph::Function>& function, bool release_function)
    : runtime::ExternalFunction(function, release_function)
    , m_function(function)
{
}

void runtime::gpu::ExternalFunction::compile()
{
}

shared_ptr<runtime::CallFrame> runtime::gpu::ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }

    return make_shared<runtime::gpu::GPUCallFrame>(shared_from_this(), m_function);
}
