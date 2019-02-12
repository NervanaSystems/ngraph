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

#include <tbb/tbb_stddef.h>

#include "cpu_backend_visibility.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

extern "C" CPU_BACKEND_API runtime::Backend* new_backend(const char* configuration_string)
{
    // Force TBB to link to the backend
    tbb::TBB_runtime_interface_version();
    return new runtime::cpu::CPU_Backend();
}

extern "C" CPU_BACKEND_API void delete_backend(runtime::Backend* backend)
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

shared_ptr<runtime::Tensor>
    runtime::cpu::CPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape, this);
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape, memory_pointer, this);
}

shared_ptr<runtime::Executable>
    runtime::cpu::CPU_Backend::compile(shared_ptr<Function> func, bool performance_counters_enabled)
{
    shared_ptr<runtime::Executable> rc;
    auto it = m_exec_map.find(func);
    if (it != m_exec_map.end())
    {
        rc = it->second;
    }
    else
    {
        rc = make_shared<CPU_Executable>(func, performance_counters_enabled);
        m_exec_map.insert({func, rc});
    }
    return rc;
}

runtime::cpu::CPU_Executable::CPU_Executable(shared_ptr<Function> func,
                                             bool performance_counters_enabled)
{
    FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function == nullptr)
    {
        instance.m_external_function = make_shared<CPU_ExternalFunction>(func);
        instance.m_external_function->m_emit_timing = performance_counters_enabled;
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<CPU_CallFrame>(cf);
    }
    set_parameters_and_results(*func);
}

std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame> runtime::cpu::CPU_Executable::get_call_frame()
{
    FunctionInstance& instance = m_function_instance;
    return instance.m_call_frame;
}

bool runtime::cpu::CPU_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    bool rc = true;

    FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function == nullptr)
    {
        NGRAPH_INFO;
        throw runtime_error("compile() must be called before call().");
    }

    instance.m_call_frame->call(outputs, inputs);

    return rc;
}

void runtime::cpu::CPU_Backend::remove_compiled_function(shared_ptr<Executable> exec)
{
    for (auto it = m_exec_map.begin(); it != m_exec_map.end(); ++it)
    {
        if (it->second == exec)
        {
            m_exec_map.erase(it);
            break;
        }
    }
}

vector<runtime::PerformanceCounter> runtime::cpu::CPU_Executable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function != nullptr)
    {
        rc.insert(rc.end(),
                  instance.m_external_function->get_perf_counters().begin(),
                  instance.m_external_function->get_perf_counters().end());
    }
    return rc;
}

bool runtime::cpu::CPU_Backend::is_supported(const Node& op) const
{
    return true;
}
bool runtime::cpu::CPU_Backend::is_supported_property(const Property prop) const
{
    if (prop == Property::memory_attach)
    {
        return true;
    }

    return false;
}
