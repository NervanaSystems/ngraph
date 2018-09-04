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

#include <tbb/tbb_stddef.h>

#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    // Force TBB to link to the backend
    tbb::TBB_runtime_interface_version();
    return new runtime::cpu::CPU_Backend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

namespace
{
    static class CPUStaticInit
    {
    public:
        CPUStaticInit() { runtime::BackendManager::register_backend("CPU", new_backend); }
        ~CPUStaticInit() {}
    } s_cpu_static_init;
}

shared_ptr<runtime::cpu::CPU_CallFrame> runtime::cpu::CPU_Backend::make_call_frame(
    const shared_ptr<runtime::cpu::CPU_ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

shared_ptr<runtime::TensorView>
    runtime::cpu::CPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
}

shared_ptr<runtime::TensorView> runtime::cpu::CPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape, memory_pointer);
}

bool runtime::cpu::CPU_Backend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function == nullptr)
    {
        instance.m_external_function = make_shared<CPU_ExternalFunction>(func);
#if !defined(NGRAPH_DEX_ONLY)
        instance.m_external_function->m_emit_timing = instance.m_performance_counters_enabled;
#endif
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<CPU_CallFrame>(cf);
    }
    return true;
}

bool runtime::cpu::CPU_Backend::call(shared_ptr<Function> func,
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

void runtime::cpu::CPU_Backend::remove_compiled_function(shared_ptr<Function> func)
{
    m_function_map.erase(func);
}

#if !defined(NGRAPH_DEX_ONLY)

void runtime::cpu::CPU_Backend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    if (instance.m_external_function != nullptr)
    {
        throw runtime_error("Performance data collection must be enabled prior to compiling.");
    }
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::cpu::CPU_Backend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    auto it = m_function_map.find(func);
    if (it != m_function_map.end())
    {
        const FunctionInstance& instance = it->second;
        if (instance.m_external_function != nullptr)
        {
            auto* engine = instance.m_external_function->m_execution_engine.get();
            if (engine)
            {
                auto get_count = engine->find_function<size_t()>("get_debug_timer_count");
                auto get_name = engine->find_function<const char*(size_t)>("get_debug_timer_name");
                auto get_microseconds =
                    engine->find_function<size_t(size_t)>("get_debug_timer_microseconds");
                auto get_call_count =
                    engine->find_function<size_t(size_t)>("get_debug_timer_call_count");

                if (get_count && get_name && get_microseconds && get_call_count)
                {
                    size_t count = get_count();
                    for (size_t i = 0; i < count; i++)
                    {
                        rc.push_back({get_name(i), get_microseconds(i), get_call_count(i)});
                    }
                }
            }
        }
    }
    return rc;
}

#endif
