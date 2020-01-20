//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/cpu/cpu_debug_tracer.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_DebugTracer::CPU_DebugTracer()
    : m_serial_number(0)
{
    static const auto debug_t = std::getenv("NGRAPH_CPU_DEBUG_TRACER");
    if (debug_t != nullptr)
    {
        m_enable_tracing = true;

        init_streams();
    }
}

void runtime::cpu::CPU_DebugTracer::init_streams()
{
    if (m_tracer_stream.is_open())
    {
        return;
    }

    static auto trace_file_path = std::getenv("NGRAPH_CPU_TRACER_LOG");
    static auto trace_bin_file_path = std::getenv("NGRAPH_CPU_BIN_TRACER_LOG");
    if (trace_file_path == nullptr)
    {
        trace_file_path = const_cast<char*>("trace_meta.log");
    }
    if (trace_bin_file_path == nullptr)
    {
        trace_bin_file_path = const_cast<char*>("trace_bin_data.log");
    }

    m_tracer_stream.open(trace_file_path, ios_base::out | ios_base::ate);
    m_tracer_bin_stream.open(trace_bin_file_path, std::ios_base::out | std::ios_base::ate);
}

void runtime::cpu::CPU_DebugTracer::set_enable_tracing(bool new_state)
{
    if (!m_enable_tracing && new_state)
    {
        init_streams();
    }

    m_enable_tracing = new_state;
}

void runtime::cpu::CPU_DebugTracer::end_of_kernel()
{
    m_serial_number++;

    m_tracer_stream.flush();
    m_tracer_bin_stream.flush();
}
