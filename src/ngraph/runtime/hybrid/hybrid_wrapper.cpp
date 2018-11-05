//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/hybrid/hybrid_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::hybrid::HybridWrapper::HybridWrapper(const string& backend_name,
                                              const set<string>& supported_ops,
                                              const string& name)
    : m_backend{runtime::Backend::create(backend_name)}
    , m_supported_ops{supported_ops}
    , m_name{name}
{
}

shared_ptr<runtime::Tensor>
    runtime::hybrid::HybridWrapper::create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
{
    return m_backend->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HybridWrapper::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return m_backend->create_tensor(element_type, shape, memory_pointer);
}

bool runtime::hybrid::HybridWrapper::compile(shared_ptr<Function> func)
{
    return m_backend->compile(func);
}

bool runtime::hybrid::HybridWrapper::call(shared_ptr<Function> func,
                                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    return m_backend->call(func, outputs, inputs);
}

bool runtime::hybrid::HybridWrapper::is_supported(const Node& node) const
{
    return m_supported_ops.find(node.description()) != m_supported_ops.end();
}
