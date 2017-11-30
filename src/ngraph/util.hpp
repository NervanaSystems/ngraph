// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace ngraph
{
    class Node;
    class Function;
    class stopwatch;
    extern std::map<std::string, stopwatch*> stopwatch_statistics;

    template <typename T>
    std::string join(const T& v, const std::string& sep = ", ")
    {
        std::ostringstream ss;
        for (const auto& x : v)
        {
            if (&x != &*(v.begin()))
            {
                ss << sep;
            }
            ss << x;
        }
        return ss.str();
    }

    template <typename T>
    static std::string vector_to_string(const std::vector<T>& v)
    {
        std::ostringstream os;
        os << "[ " << ngraph::join(v) << " ]";
        return os.str();
    }

    template <typename U, typename T>
    bool contains(const U& container, const T& obj)
    {
        bool rc = false;
        for (auto o : container)
        {
            if (o == obj)
            {
                rc = true;
                break;
            }
        }
        return rc;
    }

    template <typename U, typename T>
    bool contains_key(const U& container, const T& obj)
    {
        bool rc = false;
        for (auto o : container)
        {
            if (o.first == obj)
            {
                rc = true;
                break;
            }
        }
        return rc;
    }

    template <typename U, typename T>
    void remove_from(U& container, const T& obj)
    {
        auto it = container.find(obj);
        if (it != container.end())
        {
            container.erase(it);
        }
    }

    size_t hash_combine(const std::vector<size_t>& list);
    void dump(std::ostream& out, const void*, size_t);

    std::string to_lower(const std::string& s);
    std::string trim(const std::string& s);
    std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

    class stopwatch
    {
    public:
        stopwatch() {}
        stopwatch(const std::string& name)
            : m_name{name}
        {
            stopwatch_statistics.insert({m_name, this});
        }

        ~stopwatch()
        {
            if (m_name.size() > 0)
            {
                stopwatch_statistics.find(m_name);
            }
        }

        void start()
        {
            if (m_active == false)
            {
                m_total_count++;
                m_active = true;
                m_start_time = m_clock.now();
            }
        }

        void stop()
        {
            if (m_active == true)
            {
                auto end_time = m_clock.now();
                m_last_time = end_time - m_start_time;
                m_total_time += m_last_time;
                m_active = false;
            }
        }

        size_t get_call_count() const { return m_total_count; }
        size_t get_seconds() const { return get_nanoseconds() / 1e9; }
        size_t get_milliseconds() const { return get_nanoseconds() / 1e6; }
        size_t get_microseconds() const { return get_nanoseconds() / 1e3; }
        size_t get_nanoseconds() const
        {
            if (m_active)
            {
                return (m_clock.now() - m_start_time).count();
            }
            else
            {
                return m_last_time.count();
            }
        }

        size_t get_total_seconds() const { return get_total_nanoseconds() / 1e9; }
        size_t get_total_milliseconds() const { return get_total_nanoseconds() / 1e6; }
        size_t get_total_microseconds() const { return get_total_nanoseconds() / 1e3; }
        size_t get_total_nanoseconds() const { return m_total_time.count(); }
    private:
        std::chrono::high_resolution_clock m_clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
        bool m_active = false;
        std::chrono::nanoseconds m_total_time =
            std::chrono::high_resolution_clock::duration::zero();
        std::chrono::nanoseconds m_last_time;
        size_t m_total_count = 0;
        std::string m_name;
    };

    /// Parses a string containing a literal of the underlying type.
    template <typename T>
    T parse_string(const std::string& s)
    {
        T result;
        std::stringstream ss;

        ss << s;
        ss >> result;

        // Check that (1) parsing succeeded and (2) the entire string was used.
        if (ss.fail() || ss.rdbuf()->in_avail() != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }

        return result;
    }

    /// Parses a list of strings containing literals of the underlying type.
    template <typename T>
    std::vector<T> parse_string(const std::vector<std::string>& ss)
    {
        std::vector<T> result;

        for (auto s : ss)
        {
            result.push_back(parse_string<T>(s));
        }

        return result;
    }

    template <class InputIt, class BinaryOp>
    typename std::iterator_traits<InputIt>::value_type
        reduce(InputIt first, InputIt last, BinaryOp op)
    {
        typename std::iterator_traits<InputIt>::value_type result;

        if (first == last)
        {
            result = {};
        }
        else
        {
            result = *first++;
            while (first != last)
            {
                result = op(result, *first);
                first++;
            }
        }
        return result;
    }

    template <typename T>
    T plus(const T& a, const T& b)
    {
        return a + b;
    }

    template <typename T>
    T mul(const T& a, const T& b)
    {
        return a * b;
    }

    void traverse_nodes(Function* p, std::function<void(std::shared_ptr<Node>)> f);
    void traverse_nodes(std::shared_ptr<Function> p, std::function<void(std::shared_ptr<Node>)> f);
    void traverse_functions(std::shared_ptr<Function> p,
                            std::function<void(std::shared_ptr<Function>)> f);

    void free_nodes(std::shared_ptr<Function>);

    void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

    void replace_node_users_arguments(std::shared_ptr<Node> target,
                                      std::shared_ptr<Node> replacement);
} // end namespace ngraph
