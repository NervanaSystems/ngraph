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

#pragma once

#include <deque>
#include <sstream>
#include <stdexcept>

namespace ngraph
{
    class conststring
    {
    public:
        template <size_t SIZE>
        constexpr conststring(const char (&p)[SIZE])
            : _string(p)
            , _size(SIZE)
        {
        }

        constexpr char operator[](size_t i) const
        {
            return i < _size ? _string[i] : throw std::out_of_range("");
        }
        constexpr const char* get_ptr(size_t offset) const { return &_string[offset]; }
        constexpr size_t size() const { return _size; }
    private:
        const char* _string;
        size_t _size;
    };

    constexpr const char* find_last(conststring s, size_t offset, char ch)
    {
        return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1)
                                                             : find_last(s, offset - 1, ch));
    }

    constexpr const char* find_last(conststring s, char ch)
    {
        return find_last(s, s.size() - 1, ch);
    }
    constexpr const char* get_file_name(conststring s) { return find_last(s, '/'); }
    enum class LOG_TYPE
    {
        _LOG_TYPE_ERROR,
        _LOG_TYPE_WARNING,
        _LOG_TYPE_INFO,
        _LOG_TYPE_DEBUG,
    };

    class log_helper
    {
    public:
        log_helper(LOG_TYPE, const char* file, int line, const char* func);
        ~log_helper();

        std::ostream& stream() { return _stream; }
    private:
        std::stringstream _stream;
    };

    class logger
    {
        friend class log_helper;

    public:
        static void set_log_path(const std::string& path);
        static void start();
        static void stop();

    private:
        static void log_item(const std::string& s);
        static void process_event(const std::string& s);
        static void thread_entry(void* param);
        static std::string log_path;
        static std::deque<std::string> queue;
    };

    extern std::ostream& get_nil_stream();

#define NGRAPH_ERR                                                                                 \
    ngraph::log_helper(ngraph::LOG_TYPE::_LOG_TYPE_ERROR,                                          \
                       ngraph::get_file_name(__FILE__),                                            \
                       __LINE__,                                                                   \
                       __PRETTY_FUNCTION__)                                                        \
        .stream()
#define NGRAPH_WARN                                                                                \
    ngraph::log_helper(ngraph::LOG_TYPE::_LOG_TYPE_WARNING,                                        \
                       ngraph::get_file_name(__FILE__),                                            \
                       __LINE__,                                                                   \
                       __PRETTY_FUNCTION__)                                                        \
        .stream()
#define NGRAPH_INFO                                                                                \
    ngraph::log_helper(ngraph::LOG_TYPE::_LOG_TYPE_INFO,                                           \
                       ngraph::get_file_name(__FILE__),                                            \
                       __LINE__,                                                                   \
                       __PRETTY_FUNCTION__)                                                        \
        .stream()

#ifdef NGRAPH_DEBUG_ENABLE
#define NGRAPH_DEBUG                                                                               \
    ngraph::log_helper(ngraph::LOG_TYPE::_LOG_TYPE_DEBUG,                                          \
                       ngraph::get_file_name(__FILE__),                                            \
                       __LINE__,                                                                   \
                       __PRETTY_FUNCTION__)                                                        \
        .stream()
#else
#define NGRAPH_DEBUG ngraph::get_nil_stream()
#endif
}
