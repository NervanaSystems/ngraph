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
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_executable.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

// namespace
// {
//     static class CPUStaticInit
//     {
//     public:
//         CPUStaticInit() { runtime::BackendManager::register_backend("CPU", new_backend); }
//         ~CPUStaticInit() {}
//     } s_cpu_static_init;
// }

runtime::cpu::CPUExecutable::CPUExecutable(Backend* backend,
                                           shared_ptr<Function> func,
                                           bool enable_performance_collection)
    : Executable(backend)
{
    m_external_function = make_shared<CPU_ExternalFunction>(func);
    m_external_function->m_emit_timing = m_performance_counters_enabled;
    auto cf = m_external_function->make_call_frame();
    m_call_frame = dynamic_pointer_cast<CPU_CallFrame>(cf);

    set_parameters_and_results(*func);
}

std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame> runtime::cpu::CPUExecutable::get_call_frame()
{
    return m_call_frame;
}

bool runtime::cpu::CPUExecutable::execute(const std::vector<runtime::Tensor*>& outputs,
                                          const std::vector<runtime::Tensor*>& inputs)
{
    bool rc = true;

    m_call_frame->call(outputs, inputs);

    return rc;
}

vector<runtime::PerformanceCounter> runtime::cpu::CPUExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    if (m_external_function != nullptr)
    {
        rc.insert(rc.end(),
                  m_external_function->get_perf_counters().begin(),
                  m_external_function->get_perf_counters().end());
    }
    return rc;
}
