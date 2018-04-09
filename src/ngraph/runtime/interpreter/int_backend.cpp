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
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/int_call_frame.hpp"
#include "ngraph/runtime/interpreter/int_external_function.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::CallFrame> runtime::interpreter::INT_Backend::make_call_frame(
    const shared_ptr<runtime::ExternalFunction>& external_function)
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
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    auto rc = make_shared<runtime::HostTensorView>(element_type, shape, memory_pointer, "external");
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<ngraph::runtime::TensorView>
    runtime::interpreter::INT_Backend::create_tensor(const ngraph::element::Type& element_type,
                                                     const Shape& shape)
{
    auto rc = make_shared<runtime::HostTensorView>(element_type, shape, "external");
    return static_pointer_cast<runtime::TensorView>(rc);
}

bool runtime::interpreter::INT_Backend::compile(const ngraph::Function& func)
{
    if (!contains_key(m_function_map, &func))
    {
        FunctionInstance instance;
        instance.m_function = clone_function(func);
        instance.m_external_function =
            make_shared<interpreter::ExternalFunction>(instance.m_function);
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<interpreter::INT_CallFrame>(cf);
        m_function_map.insert({&func, instance});
    }
    return true;
}

bool runtime::interpreter::INT_Backend::call(const Function& fun,
                                             const vector<shared_ptr<runtime::TensorView>>& outputs,
                                             const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    bool rc = true;
    auto it = m_function_map.find(&fun);
    if (it == m_function_map.end())
    {
        compile(fun);
        it = m_function_map.find(&fun);
    }

    if (it == m_function_map.end())
    {
        throw runtime_error("Error constructing backend.");
    }

    FunctionInstance& instance = it->second;
    instance.m_call_frame->call(outputs, inputs);

    return rc;
}

bool runtime::interpreter::INT_Backend::call(const vector<shared_ptr<runtime::TensorView>>& outputs,
                                             const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    if (m_function_map.size() != 1)
    {
        throw runtime_error("This call method only works if a single function is compiled");
    }
    FunctionInstance& instance = m_function_map.begin()->second;
    instance.m_call_frame->call(outputs, inputs);
    return true;
}
