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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/hybrid/hybrid_executable.hpp"
#include "ngraph/runtime/hybrid/hybrid_tensor.hpp"

using namespace ngraph;
using namespace std;

runtime::hybrid::HybridBackend::HybridBackend(
    const std::vector<std::shared_ptr<runtime::Backend>>& backend_list)
    : m_backend_list{backend_list}
{
}

shared_ptr<runtime::Tensor>
    runtime::hybrid::HybridBackend::create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
{
    return m_backend_list[0]->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HybridBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return m_backend_list[0]->create_tensor(element_type, shape, memory_pointer);
}

shared_ptr<runtime::Executable>
    runtime::hybrid::HybridBackend::compile(shared_ptr<Function> func,
                                            bool enable_performance_collection)
{
    return make_shared<HybridExecutable>(
        m_backend_list, func, enable_performance_collection, m_debug_enabled);
}

bool runtime::hybrid::HybridBackend::is_supported(const Node& node) const
{
    return true;
}
