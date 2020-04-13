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
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief Interval arithmetic
    ///
    /// An interval is the set of integers from m_min_val through m_max_val.
    /// The value s_max acts like infinity. The
    /// addition, subtraction, or multiplication of intervals is the smallest interval
    /// containing the sums, differences, or products of elements of the two intervals. An empty
    /// interval is canonicalized to [s_max, s_max].
    class NGRAPH_API Interval
    {
    public:
        using value_type = std::int64_t;
        using size_type = std::uint64_t;

        /// \brief Interval of everything
        Interval() = default;
        /// \brief Copy constructor
        Interval(const Interval& interval) = default;

        /// \brief Closed interval {x|min_val <= x <= max_val}
        Interval(value_type min_val, value_type max_val);

        /// \brief Single-valued interval; just contains val
        Interval(value_type val);

        Interval& operator=(const Interval& interval) = default;

        /// \brief The number of elements in the interval. Zero if max < min.
        size_type size() const;
        /// \brief Returns true if the interval has no elements
        bool empty() const;
        /// \brief the inclusive lower bound of the interval
        value_type get_min_val() const { return m_min_val; }
        /// \brief Set the inclusive lower bound of the interval
        void set_min_val(value_type val) { m_min_val = val; }
        /// \brief the inclusive upper bound of the interval
        value_type get_max_val() const { return m_max_val; }
        /// \brief Set the inclusive upper bound of the interval
        void set_max_val(value_type val) { m_max_val = val; }
        /// \brief True if the upper bound is finite
        bool has_upper_bound() const { return m_max_val != s_max; }
        /// \brief True if min and max bounds match
        bool operator==(const Interval& interval) const;
        bool operator!=(const Interval& interval) const;

        /// \brief The interval whose elements are a sum of an element from each interval
        Interval operator+(const Interval& interval) const;

        /// \brief Extend this interval to sums of elements in this interval and interval
        Interval& operator+=(const Interval& interval);

        /// \brief The interval whose elements are a difference of an element from each interval
        Interval operator-(const Interval& interval) const;

        /// \brief Extend this interval to differences of elements in this interval and interval
        Interval& operator-=(const Interval& interval);

        /// \brief The smallest interval whose elements are a product of an element from each
        /// interval
        Interval operator*(const Interval& interval) const;

        /// \brief Extend this interval to products of elements in this interval and interval
        Interval& operator*=(const Interval& interval);

        /// \brief The interval that is the intersection of this interval and interval
        Interval operator&(const Interval& interval) const;

        /// \brief Change this interval to only include elements also in interval
        Interval& operator&=(const Interval& interval);

        /// \brief True if this interval includes value
        bool contains(value_type value) const;
        /// \brief True if this interval includes all the values in interval
        bool contains(const Interval& interval) const;

        /// \brief The value used for no upper bound
        static constexpr value_type s_max{std::numeric_limits<value_type>::max()};

    protected:
        void canonicalize();
        static value_type clip(value_type value);
        static value_type clip_times(value_type a, value_type b);
        static value_type clip_add(value_type a, value_type b);
        static value_type clip_minus(value_type a, value_type b);

        value_type m_min_val{0};
        value_type m_max_val{s_max};
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& str, const Interval& interval);
}
