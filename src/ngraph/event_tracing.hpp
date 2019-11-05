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

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
// windows.h must be before processthreadsapi.h so we need this comment
#include <processthreadsapi.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif
#include <functional>
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    //
    // This class records timestamps for a given user defined event and
    // produces output in the chrome tracing format that can be used to view
    // the events of a running program
    //
    // Following is the format of a trace event
    //
    // {
    //   "name": "myName",
    //   "cat": "category,list",
    //   "ph": "B",
    //   "ts": 12345,
    //   "pid": 123,
    //   "tid": 456,
    //   "args": {
    //     "someArg": 1,
    //     "anotherArg": {
    //       "value": "my value"
    //     }
    //   }
    // }
    //
    // The trace file format is defined here:
    // https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    //
    // The trace file can be viewed by Chrome browser using the
    // URL: chrome://tracing/
    //
    // More information about this is at:
    // http://dev.chromium.org/developers/how-tos/trace-event-profiling-tool

    class Event
    {
    public:
        explicit Event(const std::string& name,
                       const std::string& category,
                       const std::string& args)
            : m_pid(getpid())
            , m_start(std::chrono::high_resolution_clock::now())
            , m_stopped(false)
            , m_name(name)
            , m_category(category)
            , m_args(args)
        {
            m_stop = m_start;
        }

        void Stop()
        {
            if (m_stopped)
            {
                return;
            }
            m_stopped = true;
            m_stop = std::chrono::high_resolution_clock::now();
        }

        const std::string& get_name() const { return m_name; }
        const std::string& get_category() const { return m_category; }
        const std::string& get_agrs() const { return m_args; }
        const std::chrono::time_point<std::chrono::high_resolution_clock>& get_start() const
        {
            return m_start;
        }
        const std::chrono::time_point<std::chrono::high_resolution_clock>& get_stop() const
        {
            return m_stop;
        }

        static void register_event_writer(std::function<void(const Event& event)> callback)
        {
            std::lock_guard<std::mutex> lock(s_file_mutex);

            s_event_writer_registered = true;
            s_event_writer = callback;
        }
        static void write_trace(const Event& event);
        static bool is_tracing_enabled() { return s_tracing_enabled; }
        static void enable_event_tracing();
        static void disable_event_tracing();
        std::string to_json() const;

    private:
        int m_pid;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_stop;
        bool m_stopped;
        std::string m_name;
        std::string m_category;
        std::string m_args;

        NGRAPH_API static std::mutex s_file_mutex;
        NGRAPH_API static std::ofstream s_event_log;
        NGRAPH_API static bool s_tracing_enabled;
        NGRAPH_API static std::function<void(const Event& event)> s_event_writer;
        NGRAPH_API static bool s_event_writer_registered;
    };

} // namespace ngraph
