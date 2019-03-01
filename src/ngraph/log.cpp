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
using namespace std::chrono;

void ngraph::default_logger_handler_func(const string& s)
{
    cout << s + "\n";
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
    if (tm)
    {
        char buffer[256];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
        m_stream << buffer << " ";
    }

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

#if defined(__linux) || defined(__APPLE__)
std::string ngraph::get_timestamp()
{
    // get current time
    auto now = system_clock::now();

    // get number of nanoseconds for the current second
    // (remainder after division into seconds)
    auto ns = duration_cast<nanoseconds>(now.time_since_epoch()) % 1000000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream timestamp;
    timestamp << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    timestamp << '.' << std::setfill('0') << std::setw(3) << ns.count();

    return timestamp.str();
}

void ngraph::LogPrintf(const char* fmt, ...)
{
    va_list args1;
    va_start(args1, fmt);
    va_list args2;
    va_copy(args2, args1);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, fmt, args1));
    va_end(args1);
    std::vsnprintf(buf.data(), buf.size(), fmt, args2);
#pragma GCC diagnostic pop
    va_end(args2);

#ifdef NGRAPH_DISTRIBUTED_OMPI_ENABLE
    ngraph::Distributed dist;
    std::printf("%s [RANK: %d]: %s\n", get_timestamp().c_str(), dist.get_rank(), buf.data());
#else
    std::printf("%s %s\n", get_timestamp().c_str(), buf.data());
#endif
}

// This function will be executed only once during startup (loading of the DSO)
static bool CheckLoggingLevel()
{
    if (std::getenv("NGRAPH_DISABLE_LOGGING") != nullptr)
    {
        return true;
    }
    return false;
}
bool ngraph::DISABLE_LOGGING = CheckLoggingLevel();
#endif
