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

#include <limits>
#include <stddef.h>
#include <stdexcept>

namespace ngraph
{
    /// \brief Class representing a dimension, which may be dynamic (undetermined until runtime),
    ///        in a shape or shape-like object.
    ///
    /// Static dimensions may be implicitly converted from size_t. A dynamic dimension is
    /// constructed with Dimension() or Dimension::dynamic().
    ///
    /// XXX: THIS CLASS IS NOT IN USE YET AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    class Dimension
    {
    public:
        /// \brief Construct a static dimension.
        /// \param dimension Value of the dimension. Must not be equal to
        ///                  Dimension::s_dynamic_val.
        /// \throws std::invalid_argument If `dimension` == Dimension::s_dynamic_val.
        Dimension(size_t dimension);

        /// \brief Construct a dynamic dimension.
        Dimension() { m_dimension = s_dynamic_val; }
        /// \brief Check whether this dimension is static.
        /// \return `true` if the dimension is static, else `false`.
        bool is_static() const { return m_dimension != s_dynamic_val; }
        /// \brief Check whether this dimension is dynamic.
        /// \return `false` if the dimension is static, else `true`.
        bool is_dynamic() const { return !is_static(); }
        /// \brief Convert this dimension to `size_t`. This dimension must be static.
        /// \throws std::invalid_argument If this dimension is dynamic.
        explicit operator size_t() const
        {
            if (is_dynamic())
            {
                throw std::invalid_argument("Cannot convert dynamic dimension to size_t");
            }
            return m_dimension;
        }

        /// \brief Check whether this dimension represents the same scheme as the argument (both
        ///        dynamic, or equal).
        /// \param dim The other dimension to compare this dimension to.
        /// \return `true` if this dimension and `dim` are both dynamic, or if they are both
        ///         static and equal; otherwise, `false`.
        bool same_scheme(const Dimension& dim) const
        {
            return (is_dynamic() && dim.is_dynamic()) ||
                   (is_static() && dim.is_static() && m_dimension == size_t(dim));
        }

        /// \brief Try to merge two Dimension objects together.
        /// \param[out] dst Reference to write the merged Dimension into.
        /// \param d1 First dimension to merge.
        /// \param d2 Second dimension to merge.
        /// \return `true` if merging succeeds, else `false`.
        ///
        /// \li If `d1` is dynamic, writes `d2` to `dst` and returns `true`.
        /// \li If `d2` is dynamic, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are static and equal, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are both static and unequal, leaves `dst` unchanged and
        ///     returns `false`.
        static bool merge(Dimension& dst, const Dimension d1, const Dimension d2);

        /// \brief Check whether this dimension is capable of being merged with the argument
        ///        dimension.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension is compatible with `d`, else `false`.
        ///
        /// Two dimensions are considered compatible if it is possible to merge them. (See
        /// Dimension::merge.)
        bool compatible(const Dimension& d) const;
        /// \brief Create a dynamic dimension.
        /// \return A dynamic dimension.
        static Dimension dynamic() { return Dimension(); }
        /// \brief Constant for the value used internally to represent a dynamic dimension.
        static const size_t s_dynamic_val{std::numeric_limits<size_t>::max()};

        friend Dimension operator+(const Dimension& d1, const Dimension& d2);
        friend Dimension operator*(const Dimension& d1, const Dimension& d2);

        Dimension& operator+=(const Dimension& d1)
        {
            Dimension cur_val = *this;
            *this = cur_val + d1;
            return *this;
        }

        Dimension& operator*=(const Dimension& d1)
        {
            Dimension cur_val = *this;
            *this = cur_val * d1;
            return *this;
        }

    private:
        // The actual numerical value of the dimension. s_dynamic_val is a special case,
        // representing a dynamic dimension.
        size_t m_dimension;
    };

    /// \brief Insert a human-readable representation of a dimension into an output stream.
    /// \param str The output stream targeted for insertion.
    /// \param dimension The dimension to be inserted into `str`.
    /// \return A reference to `str` after insertion.
    ///
    /// Inserts the string `?` if `dimension` is dynamic; else inserts `size_t(dimension)`.
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);

    /// \brief Addition operator for dimensions.
    /// \param d1 Left operand for addition.
    /// \param d2 Right operand for addition.
    /// \return Dimension::dynamic() if either of `d1` or `d2` is dynamic; else, a static
    ///         dimension with value `size_t(d1)+size_t(d2)`.
    Dimension operator+(const Dimension& d1, const Dimension& d2);

    /// \brief Multiplication operator for dimensions.
    /// \param d1 Left operand for multiplication.
    /// \param d2 Right operand for multiplicaiton.
    /// \return 0 if either of `d1` or `d2` is static and 0; else, Dimension::dynamic() if either
    ///         of `d1` or `d2` is dynamic; else, a static dimension with value
    ///         `size_t(d1)*size_t(d2)`.
    Dimension operator*(const Dimension& d1, const Dimension& d2);
}
