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

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    class Node;
    class Function;
    class NodeMap;
    class stopwatch;

    namespace runtime
    {
        class Backend;
        class Value;
    }

    std::string to_cplusplus_sourcecode_literal(bool val);

    template <typename T>
    std::string join(const T& v, const std::string& sep = ", ")
    {
        std::ostringstream ss;
        size_t count = 0;
        for (const auto& x : v)
        {
            if (count++ > 0)
            {
                ss << sep;
            }
            ss << x;
        }
        return ss.str();
    }

    template <typename T>
    std::string vector_to_string(const T& v)
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

    size_t hash_combine(const std::vector<size_t>& list);
    void dump(std::ostream& out, const void*, size_t);

    std::string to_lower(const std::string& s);
    std::string to_upper(const std::string& s);
    std::string trim(const std::string& s);
    std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

    class stopwatch
    {
    public:
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

        size_t get_call_count() const;
        size_t get_seconds() const;
        size_t get_milliseconds() const;
        size_t get_microseconds() const;
        std::chrono::nanoseconds get_timer_value() const;
        size_t get_nanoseconds() const;

        size_t get_total_seconds() const;
        size_t get_total_milliseconds() const;
        size_t get_total_microseconds() const;
        size_t get_total_nanoseconds() const;

    private:
        std::chrono::high_resolution_clock m_clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
        bool m_active = false;
        std::chrono::nanoseconds m_total_time =
            std::chrono::high_resolution_clock::duration::zero();
        std::chrono::nanoseconds m_last_time = std::chrono::high_resolution_clock::duration::zero();
        size_t m_total_count = 0;
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

    /// template specializations for float and double to handle INFINITY, -INFINITY
    /// and NaN values.
    template <>
    float parse_string<float>(const std::string& s);
    template <>
    double parse_string<double>(const std::string& s);

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

    template <typename T>
    T ceil_div(const T& x, const T& y)
    {
        return (x == 0 ? 0 : (1 + (x - 1) / y));
    }

    template <typename T>
    T subtract_or_zero(T x, T y)
    {
        return y > x ? 0 : x - y;
    }

    void check_fp_values_isinf(const char* name, const float* array, size_t n);
    void check_fp_values_isinf(const char* name, const double* array, size_t n);
    void check_fp_values_isnan(const char* name, const float* array, size_t n);
    void check_fp_values_isnan(const char* name, const double* array, size_t n);

    void* aligned_alloc(size_t alignment, size_t size);
    void aligned_free(void*);
    size_t round_up(size_t size, size_t alignment);
    template <typename T>
    T apply_permutation(T input, ngraph::AxisVector order);

    AxisVector get_default_order(size_t rank);
    AxisVector get_default_order(const Shape& shape);

    /*
    * Return type struct for cache_fprop, with the modified fprop and bprop
    * functions
    * and a list of the nodes that have been appended to fprop output/bprop
    * input
    */
    struct FpropCache
    {
        std::shared_ptr<Function> fprop;
        std::shared_ptr<Function> bprop;
        std::vector<std::shared_ptr<Node>> fprop_output_nodes;
        std::shared_ptr<NodeMap> node_param_map;
    };

    /**
    * This utility takes forward-propogation and back-propagation functions
    * and turns them into clone functions where the intermediate values of
    * the forward prop are added to the output of fprop and the input of the bprop
    * to avoid repeat calcualtions.
    * The last argument is the adjoints coming into the bprop function, the output
    * bprop function will have these nodes as the first N input parameters
    **/
    FpropCache cache_fprop(std::shared_ptr<Function> fprop, std::shared_ptr<Function> bprop);
} // end namespace ngraph

std::ostream& operator<<(std::ostream& os, const ngraph::NodeVector& nv);
