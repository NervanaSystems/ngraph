//*****************************************************************************
// Copyright 2019 Intel Corporation
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
using namespace ngraph;

static bool read_tracing_env_var()
{
    return (getenv("NGRAPH_ENABLE_TRACING") != nullptr);
}

mutex event::Duration::s_file_mutex;
bool event::Duration::s_tracing_enabled = read_tracing_env_var();

void event::Duration::write_trace(const Duration& event)
{
    if (is_tracing_enabled())
    {
        lock_guard<mutex> lock(s_file_mutex);

        static bool so_initialized = false;
        if (!so_initialized)
        {
            // Open the file
            Manager::open("ngraph_event_trace.json");
            so_initialized = true;
        }
        else
        {
            Manager::get_output_stream() << ",\n";
        }

        Manager::get_output_stream() << event.to_json();
    }
}

string event::Duration::to_json() const
{
    auto start_time = m_start.count();
    auto stop_time = m_stop.count();

    string s1 = R"({"name":")" + m_name + R"(","cat":")" + m_category + R"(","ph":"X")" +
                R"(,"pid":)" + Manager::get_process_id() + R"(,"tid":)" + Manager::get_thread_id() +
                R"(,"ts":)" + to_string(start_time) + R"(,"dur":)" +
                to_string(stop_time - start_time) + "}";
    return s1;
}

void event::Duration::enable_event_tracing()
{
    s_tracing_enabled = true;
}

void event::Duration::disable_event_tracing()
{
    s_tracing_enabled = false;
}

event::Object::Object(const string& name, nlohmann::json args)
    : m_name{name}
    , m_id{static_cast<size_t>(chrono::high_resolution_clock::now().time_since_epoch().count())}
{
    ostream& out = Manager::get_output_stream();
    out << R"({"name":)" + m_name + R"(,"ph":"N")" + R"(,"id":)" + to_string(m_id) + R"(,"ts":)" +
               to_string(Manager::get_current_microseconds().count()) +
               R"(,"pid":)" + Manager::get_process_id() + R"(,"tid":)" + Manager::get_thread_id() +
               "}";
}

void event::Object::snapshot(nlohmann::json args)
{
}

void event::Object::destroy()
{
}

void event::Manager::open(const string& path)
{
    ofstream& out = get_output_stream();
    if (out.is_open() == false)
    {
        out.open("ngraph_event_trace.json", ios_base::trunc);
        out << "[\n";
    }
}

void event::Manager::close()
{
    ofstream& out = get_output_stream();
    if (out.is_open())
    {
        out << "\n]\n";
        out.close();
    }
}

ofstream& event::Manager::get_output_stream()
{
    static ofstream s_event_log;
    return s_event_log;
}

const string& event::Manager::get_process_id()
{
    static const string s_pid = to_string(getpid());
    return s_pid;
}
