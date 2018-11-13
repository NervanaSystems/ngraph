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

shared_ptr<runtime::Tensor>
    runtime::cpu::CPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape, memory_pointer);
}

runtime::Handle runtime::cpu::CPU_Backend::compile(const shared_ptr<Function>& func)
{
    auto instance = make_shared<FunctionInstance>();
    m_instances.push_back(instance);
    instance->m_performance_counters_enabled = m_performance_counters_enabled;

    instance->m_external_function = make_shared<CPU_ExternalFunction>(func);
    instance->m_external_function->m_emit_timing = instance->m_performance_counters_enabled;
    auto cf = instance->m_external_function->make_call_frame();
    instance->m_call_frame = dynamic_pointer_cast<CPU_CallFrame>(cf);

    return instance.get();
}

std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame>
    runtime::cpu::CPU_Backend::get_call_frame(runtime::Handle handle)
{
    FunctionInstance* instance = static_cast<FunctionInstance*>(handle);

    return instance->m_call_frame;
}

bool runtime::cpu::CPU_Backend::call(runtime::Handle handle,
                                     const vector<shared_ptr<runtime::Tensor>>& outputs,
                                     const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    bool rc = true;

    FunctionInstance* instance = static_cast<FunctionInstance*>(handle);

    instance->m_call_frame->call(outputs, inputs);

    return rc;
}

void runtime::cpu::CPU_Backend::remove_compiled_function(runtime::Handle handle)
{
    for (auto it = m_instances.begin(); it != m_instances.end(); ++it)
    {
        if ((*it).get() == handle)
        {
            m_instances.erase(it);
            break;
        }
    }
}

void runtime::cpu::CPU_Backend::enable_performance_data(bool enable)
{
    m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::cpu::CPU_Backend::get_performance_data(runtime::Handle handle) const
{
    vector<runtime::PerformanceCounter> rc;
    FunctionInstance* instance = static_cast<FunctionInstance*>(handle);
    if (instance->m_external_function != nullptr)
    {
        rc.insert(rc.end(),
                  instance->m_external_function->get_perf_counters().begin(),
                  instance->m_external_function->get_perf_counters().end());
    }
    return rc;
}

const op::ParameterVector& runtime::cpu::CPU_Backend::get_parameter_descriptors(Handle handle) const
{
    FunctionInstance* instance = static_cast<FunctionInstance*>(handle);
    return instance->m_external_function->m_function->get_parameters();
}

const ResultVector& runtime::cpu::CPU_Backend::get_result_descriptors(Handle handle) const
{
    FunctionInstance* instance = static_cast<FunctionInstance*>(handle);
    return instance->m_external_function->m_function->get_results();
}
