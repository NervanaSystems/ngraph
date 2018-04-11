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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/int_call_frame.hpp"
#include "ngraph/runtime/interpreter/int_external_function.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::interpreter::INT_CallFrame> runtime::interpreter::INT_Backend::make_call_frame(
    const shared_ptr<runtime::interpreter::ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

shared_ptr<runtime::TensorView>
    runtime::interpreter::INT_Backend::create_tensor(const element::Type& element_type,
                                                     const Shape& shape)
{
    return make_shared<runtime::HostTensorView>(element_type, shape, "external");
}

shared_ptr<runtime::TensorView> runtime::interpreter::INT_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::HostTensorView>(element_type, shape, memory_pointer, "external");
}

bool runtime::interpreter::INT_Backend::compile(std::shared_ptr<Function> func)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        instance.m_external_function = make_shared<ExternalFunction>(func);
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<INT_CallFrame>(cf);
        instance.m_call_frame->set_nan_check(instance.m_nan_check_enabled);
    }
    return true;
}

bool runtime::interpreter::INT_Backend::call(std::shared_ptr<Function> func,
                                             const vector<shared_ptr<runtime::TensorView>>& outputs,
                                             const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    bool rc = true;

    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        rc = compile(func);
    }

    instance.m_call_frame->call(outputs, inputs);

    return rc;
}

void runtime::interpreter::INT_Backend::set_nan_check(std::shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_nan_check_enabled = enable;
}
