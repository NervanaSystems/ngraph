//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib> // llvm 8.1 gets confused about `malloc` otherwise
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    class Node;
    class Function;
    class stopwatch;

    namespace runtime
    {
        class Backend;
        class Value;
        class Tensor;
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

    size_t hash_combine(const std::vector<size_t>& list);
    void dump(std::ostream& out, const void*, size_t);

    std::string to_lower(const std::string& s);
    std::string to_upper(const std::string& s);
    std::string trim(const std::string& s);
    std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);
    template <typename T>
    std::string locale_string(T x)
    {
        std::stringstream ss;
        ss.imbue(std::locale(""));
        ss << x;
        return ss.str();
    }

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
        std::vector<T> result(ss.size());
        std::transform(ss.begin(), ss.end(), result.begin(), [](const std::string& s) {
            return parse_string<T>(s);
        });
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

    void* ngraph_malloc(size_t size);
    void ngraph_free(void*);

    size_t round_up(size_t size, size_t alignment);
    bool is_valid_permutation(ngraph::AxisVector permutation, ngraph::Rank rank = Rank::dynamic());
    template <typename T>
    T apply_permutation(T input, ngraph::AxisVector order);

    AxisVector get_default_order(size_t rank);
    NGRAPH_API
    AxisVector get_default_order(const Shape& shape);

    AxisVector get_permutation_to_default_order(const AxisVector& axis_order);

    //
    // Return type struct for cache_fprop, with the modified fprop and bprop
    // functions
    // and a list of the nodes that have been appended to fprop output/bprop
    // input
    //
    struct FpropCache
    {
        std::shared_ptr<Function> fprop;
        std::shared_ptr<Function> bprop;
        std::vector<Node*> fprop_output_nodes;
        NodeMap node_param_map;
    };

    //
    // This utility takes forward-propogation and back-propagation functions
    // and turns them into clone functions where the intermediate values of
    // the forward prop are added to the output of fprop and the input of the bprop
    // to avoid repeat calculations.
    // The last argument is the adjoints coming into the bprop function, the output
    // bprop function will have these nodes as the first N input parameters
    //
    FpropCache cache_fprop(std::shared_ptr<Function> fprop, std::shared_ptr<Function> bprop);

    // NodeExecutors are used in compiler optimization passes like ConstantFolding to execute a node
    // using the supplied input and output memory locations.
    // A BuildNodeExecutor returns a backend-specific NodeExecutor for a given Node type
    using NodeExecutorTy =
        std::function<void(const std::vector<void*>& inputs, std::vector<void*>& outputs)>;
    using BuildNodeExecutor = std::function<NodeExecutorTy(const ngraph::Node*)>;

    using BuildNodeExecutorMap = std::unordered_map<std::type_index, BuildNodeExecutor>;

    enum class TensorRole
    {
        INPUT,
        CONSTANT,
        OUTPUT,
        INTERMEDIATE,
        UNKNOWN
    };

    //
    // EnumMask is intended to work with a scoped enum type. It's used to store
    // a combination of enum values and provides easy access and manipulation
    // of these enum values as a mask.
    //
    // EnumMask does not provide a set_all() or invert() operator because they
    // could do things unexpected by the user, i.e. for enum with 4 bit values,
    // invert(001000...) != 110100..., due to the extra bits.
    //
    template <typename T>
    class EnumMask
    {
    public:
        /// Make sure the template type is an enum.
        static_assert(std::is_enum<T>::value, "EnumMask template type must be an enum");
        /// Extract the underlying type of the enum.
        typedef typename std::underlying_type<T>::type value_type;
        /// Some bit operations are not safe for signed values, we require enum
        /// type to use unsigned underlying type.
        static_assert(std::is_unsigned<value_type>::value, "EnumMask enum must use unsigned type.");

        constexpr EnumMask()
            : m_value{0}
        {
        }
        constexpr EnumMask(const T& enum_value)
            : m_value{static_cast<value_type>(enum_value)}
        {
        }
        EnumMask(const EnumMask& other)
            : m_value{other.m_value}
        {
        }
        EnumMask(std::initializer_list<T> enum_values)
            : m_value{0}
        {
            for (auto& v : enum_values)
            {
                m_value |= static_cast<value_type>(v);
            }
        }
        value_type value() const { return m_value; }
        /// Check if any of the input parameter enum bit mask match
        bool is_any_set(const EnumMask& p) const { return m_value & p.m_value; }
        /// Check if all of the input parameter enum bit mask match
        bool is_set(const EnumMask& p) const { return (m_value & p.m_value) == p.m_value; }
        /// Check if any of the input parameter enum bit mask does not match
        bool is_any_clear(const EnumMask& p) const { return !is_set(p); }
        /// Check if all of the input parameter enum bit mask do not match
        bool is_clear(const EnumMask& p) const { return !is_any_set(p); }
        void set(const EnumMask& p) { m_value |= p.m_value; }
        void clear(const EnumMask& p) { m_value &= ~p.m_value; }
        void clear_all() { m_value = 0; }
        bool operator[](const EnumMask& p) const { return is_set(p); }
        bool operator==(const EnumMask& other) const { return m_value == other.m_value; }
        bool operator!=(const EnumMask& other) const { return m_value != other.m_value; }
        EnumMask& operator=(const EnumMask& other)
        {
            m_value = other.m_value;
            return *this;
        }
        EnumMask& operator&=(const EnumMask& other)
        {
            m_value &= other.m_value;
            return *this;
        }

        EnumMask& operator|=(const EnumMask& other)
        {
            m_value |= other.m_value;
            return *this;
        }

        EnumMask operator&(const EnumMask& other) const
        {
            return EnumMask(m_value & other.m_value);
        }

        EnumMask operator|(const EnumMask& other) const
        {
            return EnumMask(m_value | other.m_value);
        }

        friend std::ostream& operator<<(std::ostream& os, const EnumMask& m)
        {
            os << m.m_value;
            return os;
        }

    private:
        /// Only used internally
        explicit EnumMask(const value_type& value)
            : m_value{value}
        {
        }

        value_type m_value;
    };

    /// \brief Function to query parsed version information of the version of ngraph which
    /// contains this function. Version information strictly follows Semantic Versioning
    /// http://semver.org
    /// \param version The major part of the version
    /// \param major Returns the major part of the version
    /// \param minor Returns the minor part of the version
    /// \param patch Returns the patch part of the version
    /// \param extra Returns the extra part of the version. This includes everything following
    /// the patch version number.
    ///
    /// \note Throws a runtime_error if there is an error during parsing
    void parse_version_string(
        std::string version, size_t& major, size_t& minor, size_t& patch, std::string& extra);
} // end namespace ngraph

template <typename T>
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::Tensor> tv);

std::vector<float> read_float_vector(std::shared_ptr<ngraph::runtime::Tensor> tv);

std::ostream& operator<<(std::ostream& os, const ngraph::NodeVector& nv);
