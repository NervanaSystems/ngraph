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

#include "ngraph/runtime/cpu/cpu_debug_tracer.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_DebugTracer::CPU_DebugTracer()
    : m_serial_number(0)
{
}

runtime::cpu::CPU_DebugTracer& runtime::cpu::CPU_DebugTracer::getInstance()
{
    static CPU_DebugTracer instance;
    return instance;
}

void runtime::cpu::CPU_DebugTracer::init_streams(const string& trace_file_path,
                                                 const string& trace_bin_file_path)
{
    if (m_tracer_stream.is_open())
    {
        return;
    }
    m_tracer_stream.open(trace_file_path, ios_base::out | ios_base::ate);
    m_tracer_bin_stream.open(trace_bin_file_path, std::ios_base::out | std::ios_base::ate);
}

void runtime::cpu::CPU_DebugTracer::end_of_kernel()
{
    m_serial_number++;

    m_tracer_stream.flush();
    m_tracer_bin_stream.flush();
}
