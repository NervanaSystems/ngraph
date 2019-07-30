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

#include <iostream>
#include <sstream>
#include <string>

#include "event_tracing.hpp"
#include "nlohmann/json.hpp"

using namespace std;

static bool read_tracing_env_var()
{
    return (std::getenv("NGRAPH_ENABLE_TRACING") != nullptr);
}

mutex ngraph::Event::s_file_mutex;
ofstream ngraph::Event::s_event_log;
bool ngraph::Event::s_tracing_enabled = read_tracing_env_var();

void ngraph::Event::write_trace(const ngraph::Event& event)
{
    if (is_tracing_enabled())
    {
        lock_guard<mutex> lock(s_file_mutex);

        static bool so_initialized = false;
        if (!so_initialized)
        {
            // Open the file
            s_event_log.open("ngraph_event_trace.json", ios_base::trunc);
            s_event_log << "[\n";
            so_initialized = true;
        }
        else
        {
            s_event_log << ",\n";
        }

        s_event_log << event.to_json() << "\n" << flush;
    }
}

string ngraph::Event::to_json() const
{
    ostringstream thread_id;
    thread_id << this_thread::get_id();

    nlohmann::json json_start = {{"name", m_name},
                                 {"cat", m_category},
                                 {"ph", "B"},
                                 {"pid", m_pid},
                                 {"tid", thread_id.str()},
                                 {"ts", m_start.time_since_epoch().count() / 1000},
                                 {"args", m_args}};
    nlohmann::json json_end = {{"name", m_name},
                               {"cat", m_category},
                               {"ph", "E"},
                               {"pid", m_pid},
                               {"tid", thread_id.str()},
                               {"ts", m_stop.time_since_epoch().count() / 1000},
                               {"args", m_args}};
    ostringstream output;
    output << json_start << ",\n" << json_end;
    return output.str();
}

void ngraph::Event::enable_event_tracing()
{
    s_tracing_enabled = true;
}

void ngraph::Event::disable_event_tracing()
{
    s_tracing_enabled = false;
}
