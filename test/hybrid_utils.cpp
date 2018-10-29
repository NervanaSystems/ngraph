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

#include "hybrid_utils.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<runtime::Tensor> TestBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
}

shared_ptr<runtime::Tensor> TestBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape,
                                                       void* memory_pointer)
{
}

bool TestBackend::compile(shared_ptr<Function> func)
{
}

bool TestBackend::call(shared_ptr<Function> func,
                       const vector<shared_ptr<runtime::Tensor>>& outputs,
                       const vector<shared_ptr<runtime::Tensor>>& inputs)
{
}

bool TestBackend::is_supported(const Node& node) const
{
}

BackendWrapper::BackendWrapper(shared_ptr<runtime::Backend> be, set<string> supported_ops);

shared_ptr<runtime::Tensor> BackendWrapper::create_tensor(const element::Type& element_type,
                                                          const Shape& shape)
{
}

shared_ptr<runtime::Tensor> BackendWrapper::create_tensor(const element::Type& element_type,
                                                          const Shape& shape,
                                                          void* memory_pointer)
{
}

bool BackendWrapper::compile(shared_ptr<Function> func)
{
}

bool BackendWrapper::call(shared_ptr<Function> func,
                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
}

bool BackendWrapper::is_supported(const Node& node) const
{
}
