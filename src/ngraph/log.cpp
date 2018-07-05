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

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    class ThreadStarter;
}

string Logger::m_log_path;
deque<string> Logger::m_queue;
static mutex queue_mutex;
static condition_variable queue_condition;
static unique_ptr<thread> queue_thread;
static bool active = false;

class ngraph::ThreadStarter
{
public:
    ThreadStarter() { Logger::start(); }
    virtual ~ThreadStarter() { Logger::stop(); }
};

static ThreadStarter s_starter;

ostream& ngraph::get_nil_stream()
{
    static stringstream nil;
    return nil;
}

void Logger::set_log_path(const string& path)
{
    m_log_path = path;
}

void Logger::start()
{
    active = true;
    queue_thread = unique_ptr<thread>(new thread(&thread_entry, nullptr));
}

void Logger::stop()
{
    {
        unique_lock<mutex> lk(queue_mutex);
        active = false;
        queue_condition.notify_one();
    }
    queue_thread->join();
}

void Logger::process_event(const string& s)
{
    cout << s << "\n";
}

void Logger::thread_entry(void* param)
{
    unique_lock<mutex> lk(queue_mutex);
    while (active)
    {
        queue_condition.wait(lk);
        while (!m_queue.empty())
        {
            process_event(m_queue.front());
            m_queue.pop_front();
        }
    }
}

void Logger::log_item(const string& s)
{
    unique_lock<mutex> lk(queue_mutex);
    m_queue.push_back(s);
    queue_condition.notify_one();
}

void ngraph::default_logger_handler_func(const string& s)
{
    cout << s << endl;
}

LogHelper::LogHelper(LOG_TYPE type,
                     const char* file,
                     int line,
                     function<void(const string&)> handler_func)
    : m_handler_func(handler_func)
{
    switch (type)
    {
    case LOG_TYPE::_LOG_TYPE_ERROR: m_stream << "[ERR] "; break;
    case LOG_TYPE::_LOG_TYPE_WARNING: m_stream << "[WARN] "; break;
    case LOG_TYPE::_LOG_TYPE_INFO: m_stream << "[INFO] "; break;
    case LOG_TYPE::_LOG_TYPE_DEBUG: m_stream << "[DEBUG] "; break;
    }

    time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = gmtime(&tt);
    char buffer[256];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
    m_stream << buffer << " ";

    m_stream << file;
    m_stream << " " << line;
    m_stream << "\t";
}

LogHelper::~LogHelper()
{
    if (m_handler_func)
    {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
}
