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

static bool static_init()
{
    runtime::Backend::register_backend("INTERPRETER",
                                       make_shared<runtime::interpreter::INT_Backend>());
    return true;
};

bool runtime::interpreter::INT_Backend::init = static_init();

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

bool runtime::interpreter::INT_Backend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        instance.m_external_function = make_shared<ExternalFunction>(func);
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<INT_CallFrame>(cf);
        instance.m_call_frame->m_emit_timing = instance.m_performance_counters_enabled;
        instance.m_call_frame->set_nan_check(instance.m_nan_check_enabled);
    }
    return true;
}

bool runtime::interpreter::INT_Backend::call(shared_ptr<Function> func,
                                             const vector<shared_ptr<runtime::TensorView>>& outputs,
                                             const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    bool rc = true;

    validate_call(func, outputs, inputs);

    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        rc = compile(func);
    }

    instance.m_call_frame->call(outputs, inputs);

    return rc;
}

void runtime::interpreter::INT_Backend::set_nan_check(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_nan_check_enabled = enable;
}

void runtime::interpreter::INT_Backend::enable_performance_data(shared_ptr<Function> func,
                                                                bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INT_Backend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_map.at(func);
    for (const pair<const Node*, stopwatch> p : instance.m_call_frame->m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}
