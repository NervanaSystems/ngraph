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

namespace
{
    class NilStreamBuf final : public streambuf
    {
        // N.B. We derive from the base streambuf implementation, in
        //      which underflow() and overflow() both return
        //      Traits::eof() -- any access returns a failure.
    };
}

ostream& ngraph::get_nil_stream()
{
    // N.B. When debug logging is disabled, multiple threads may
    //      access the nil stream simultaneously, so it's important to
    //      return a threadsafe nil stream implementation.
    static NilStreamBuf nil_buf;
    static ostream nil{&nil_buf};
    return nil;
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
