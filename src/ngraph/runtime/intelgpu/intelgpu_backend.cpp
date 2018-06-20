/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"

using namespace std;
using namespace ngraph;

extern "C" void create_backend(void)
{
    runtime::Backend::register_backend("INTELGPU",
                                       make_shared<runtime::intelgpu::IntelGPUBackend>());
};

shared_ptr<runtime::TensorView>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    throw runtime_error("IntelGPUBackend::create_tensor: Not implemented yet");
}

shared_ptr<runtime::TensorView> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw runtime_error("IntelGPUBackend::create_tensor: Not implemented yet");
}

bool runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func)
{
    throw runtime_error("IntelGPUBackend::compile: Not implemented yet");
}

bool runtime::intelgpu::IntelGPUBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    throw runtime_error("IntelGPUBackend::call: Not implemented yet");
}
