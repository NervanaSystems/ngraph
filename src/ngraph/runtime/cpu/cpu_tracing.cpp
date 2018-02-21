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

#include <fstream>
#include <map>

#include "cpu_tracing.hpp"

ngraph::runtime::cpu::TraceEvent::TraceEvent(const std::string& ph,
                                             const std::string& cat,
                                             const std::string& name,
                                             unsigned int pid,
                                             unsigned int tid,
                                             int64_t ts,
                                             int64_t dur,
                                             const std::vector<std::string>& outputs,
                                             const std::vector<std::string>& inputs)
    : m_phase(ph)
    , m_category(cat)
    , m_name(name)
    , m_pid(pid)
    , m_tid(tid)
    , m_timestamp(ts)
    , m_duration(dur)
    , m_outputs(outputs)
    , m_inputs(inputs)
{
}

void ngraph::runtime::cpu::to_json(nlohmann::json& json, const TraceEvent& event)
{
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < event.m_inputs.size(); i++)
    {
        args["Input" + std::to_string(i + 1)] = event.m_inputs[i];
    }
    for (size_t i = 0; i < event.m_outputs.size(); i++)
    {
        args["Output" + std::to_string(i + 1)] = event.m_outputs[i];
    }

    json = nlohmann::json{{"ph", event.m_phase},
                          {"cat", event.m_category},
                          {"name", event.m_name},
                          {"pid", event.m_pid},
                          {"tid", event.m_tid},
                          {"ts", event.m_timestamp},
                          {"dur", event.m_duration},
                          {"args", args}};
}

void ngraph::runtime::cpu::generate_timeline(const std::vector<OpAttributes>& op_attrs,
                                             int64_t* op_durations)
{
    nlohmann::json timeline;
    std::list<TraceEvent> trace;
    std::ofstream out("timeline.json");

    int64_t ts = 0;
    for (size_t i = 0; i < op_attrs.size(); i++)
    {
        trace.emplace_back("X",
                           "Op",
                           op_attrs[i].m_description,
                           0,
                           0,
                           ts,
                           op_durations[i],
                           op_attrs[i].m_outputs,
                           op_attrs[i].m_inputs);
        ts += op_durations[i];
    }

    timeline["traceEvents"] = trace;
    out << timeline;
    out.close();

    return;
}

bool ngraph::runtime::cpu::is_tracing_enabled()
{
    return (std::getenv("NGRAPH_CPU_TRACING") != nullptr);
}
