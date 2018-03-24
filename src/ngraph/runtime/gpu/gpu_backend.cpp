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

#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"

using namespace ngraph;
using namespace std;

std::shared_ptr<ngraph::runtime::CallFrame> runtime::gpu::GPU_Backend::make_call_frame(
    const std::shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

std::shared_ptr<ngraph::runtime::TensorView>
    runtime::gpu::GPU_Backend::make_primary_tensor_view(const ngraph::element::Type& element_type,
                                                        const Shape& shape)
{
    auto rc = make_shared<runtime::gpu::GPU_TensorView>(element_type, shape);
    return dynamic_pointer_cast<runtime::TensorView>(rc);
}

bool runtime::interpreter::GPU_Backend::compile(const std::shared_ptr<ngraph::Function>& fun)
{
    return false;
}

bool runtime::interpreter::GPU_Backend::is_callable() const
{
    return false;
}

bool runtime::interpreter::GPU_Backend::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
    const std::vector<std::shared_ptr<runtime::TensorView>>& inputs)
{
    return false;
}
