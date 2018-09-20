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

#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace ngraph
{
    class ConstString
    {
    public:
        template <size_t SIZE>
        constexpr ConstString(const char (&p)[SIZE])
            : m_string(p)
            , m_size(SIZE)
        {
        }

        constexpr char operator[](size_t i) const
        {
            return i < m_size ? m_string[i] : throw std::out_of_range("");
        }
        constexpr const char* get_ptr(size_t offset) const { return &m_string[offset]; }
        constexpr size_t size() const { return m_size; }
    private:
        const char* m_string;
        size_t m_size;
    };

    constexpr const char* find_last(ConstString s, size_t offset, char ch)
    {
        return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1)
                                                             : find_last(s, offset - 1, ch));
    }

    constexpr const char* find_last(ConstString s, char ch)
    {
        return find_last(s, s.size() - 1, ch);
    }

    constexpr const char* get_file_name(ConstString s) { return find_last(s, '/'); }
    enum class LOG_TYPE
    {
        _LOG_TYPE_ERROR,
        _LOG_TYPE_WARNING,
        _LOG_TYPE_INFO,
        _LOG_TYPE_DEBUG,
    };

    class LogHelper
    {
    public:
        LogHelper(LOG_TYPE,
                  const char* file,
                  int line,
                  std::function<void(const std::string&)> m_handler_func);
        ~LogHelper();

        std::ostream& stream() { return m_stream; }
    private:
        std::function<void(const std::string&)> m_handler_func;
        std::stringstream m_stream;
    };

    class Logger
    {
        friend class LogHelper;

    public:
        static void set_log_path(const std::string& path);
        static void start();
        static void stop();

    private:
        static void log_item(const std::string& s);
        static void process_event(const std::string& s);
        static void thread_entry(void* param);
        static std::string m_log_path;
        static std::deque<std::string> m_queue;
    };

    extern std::ostream& get_nil_stream();

    void default_logger_handler_func(const std::string& s);

#define NGRAPH_ERR                                                                                 \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_ERROR,                                           \
                      ngraph::get_file_name(__FILE__),                                             \
                      __LINE__,                                                                    \
                      ngraph::default_logger_handler_func)                                         \
        .stream()

#define NGRAPH_WARN                                                                                \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_WARNING,                                         \
                      ngraph::get_file_name(__FILE__),                                             \
                      __LINE__,                                                                    \
                      ngraph::default_logger_handler_func)                                         \
        .stream()

#define NGRAPH_INFO                                                                                \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_INFO,                                            \
                      ngraph::get_file_name(__FILE__),                                             \
                      __LINE__,                                                                    \
                      ngraph::default_logger_handler_func)                                         \
        .stream()

#ifdef NGRAPH_DEBUG_ENABLE
#define NGRAPH_DEBUG                                                                               \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_DEBUG,                                           \
                      ngraph::get_file_name(__FILE__),                                             \
                      __LINE__,                                                                    \
                      ngraph::default_logger_handler_func)                                         \
        .stream()
#else
#define NGRAPH_DEBUG ngraph::get_nil_stream()
#endif
}
