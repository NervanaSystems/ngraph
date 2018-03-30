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

#include "ngraph/log.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::CallFrame> runtime::interpreter::INT_Backend::make_call_frame(
    const shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

shared_ptr<runtime::TensorView>
    runtime::interpreter::INT_Backend::make_primary_tensor_view(const element::Type& element_type,
                                                                const Shape& shape)
{
    auto rc = make_shared<runtime::HostTensorView>(element_type, shape, "external");
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::interpreter::INT_Backend::make_primary_tensor_view(
    const element::Type& element_type, const Shape& shape, void* mem_handle)
{
    auto rc = make_shared<runtime::HostTensorView>(element_type, shape, mem_handle, "external");
    return static_pointer_cast<runtime::TensorView>(rc);
}
