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

#pragma once

#include <limits>
#include <stddef.h>
#include <stdexcept>

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief Class representing a dimension, which may be dynamic (undetermined until runtime),
    ///        in a shape or shape-like object.
    ///
    /// Static dimensions may be implicitly converted from int64_t. A dynamic dimension is
    /// constructed with Dimension() or Dimension::dynamic().
    ///
    /// XXX: THIS CLASS IS NOT IN USE YET AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    class NGRAPH_API Dimension
    {
    public:
        /// \brief Construct a static dimension.
        /// \param dimension Value of the dimension. Must not be equal to
        ///                  Dimension::s_dynamic_val.
        /// \throws std::invalid_argument If `dimension` == Dimension::s_dynamic_val.
        Dimension(int64_t dimension);

        /// \brief Construct a dynamic dimension.
        Dimension() { m_dimension = s_dynamic_val; }
        /// \brief Check whether this dimension is static.
        /// \return `true` if the dimension is static, else `false`.
        bool is_static() const { return m_dimension != s_dynamic_val; }
        /// \brief Check whether this dimension is dynamic.
        /// \return `false` if the dimension is static, else `true`.
        bool is_dynamic() const { return !is_static(); }
        /// \brief Convert this dimension to `int64_t`. This dimension must be static.
        /// \throws std::invalid_argument If this dimension is dynamic.
        explicit operator int64_t() const
        {
            if (is_dynamic())
            {
                throw std::invalid_argument("Cannot convert dynamic dimension to int64_t");
            }
            return m_dimension;
        }
        /// \brief Convert this dimension to `size_t`. This dimension must be static and
        ///        non-negative.
        /// \throws std::invalid_argument If this dimension is dynamic or negative.
        explicit operator size_t() const
        {
            if (is_dynamic())
            {
                throw std::invalid_argument("Cannot convert dynamic dimension to size_t");
            }
            if (m_dimension < 0)
            {
                throw std::invalid_argument("Cannot convert negative dimension to size_t");
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
                   (is_static() && dim.is_static() && m_dimension == int64_t(dim));
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

        /// \brief Try to merge two Dimension objects together with implicit broadcasting
        ///        of unit-sized dimension to non unit-sized dimension
        static bool broadcast_merge(Dimension& dst, const Dimension d1, const Dimension d2);

        /// \brief Check whether this dimension is capable of being merged with the argument
        ///        dimension.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension is compatible with `d`, else `false`.
        ///
        /// Two dimensions are considered compatible if it is possible to merge them. (See
        /// Dimension::merge.)
        bool compatible(const Dimension& d) const;

        /// \brief Check whether this dimension is a relaxation of the argument.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension relaxes `d`, else `false`.
        ///
        /// A dimension `d1` _relaxes_ (or _is a relaxation of_) `d2` if `d1` and `d2` are static
        /// and equal, or `d1` is dynamic.
        ///
        /// `d1.relaxes(d2)` is equivalent to `d2.refines(d1)`.
        bool relaxes(const Dimension& d) const;

        /// \brief Check whether this dimension is a refinement of the argument.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension relaxes `d`, else `false`.
        ///
        /// A dimension `d2` _refines_ (or _is a refinement of_) `d1` if `d1` and `d2` are static
        /// and equal, or `d2` is dynamic.
        ///
        /// `d1.refines(d2)` is equivalent to `d2.relaxes(d1)`.
        bool refines(const Dimension& d) const;

        /// \brief Create a dynamic dimension.
        /// \return A dynamic dimension.
        static Dimension dynamic() { return Dimension(); }
        /// \brief Constant for the value used internally to represent a dynamic dimension.
        static const int64_t s_dynamic_val{(std::numeric_limits<int64_t>::max())};

        /// \brief Addition operator for Dimension.
        /// \param dim Right operand for addition.
        /// \return Dimension::dynamic() if either of `*this` or `dim` is dynamic; else, a static
        ///         dimension with value `int64_t(*this)+in64_t(dim)`.
        Dimension operator+(const Dimension& dim) const;

        /// \brief Subtraction operator for Dimension.
        /// \param dim Right operand for subtraction.
        /// \return Dimension::dynamic() if either of `*this` or `dim` is dynamic; else, a static
        ///         dimension with value `int64_t(*this)-int64_t(dim)`.
        Dimension operator-(const Dimension& dim) const;

        /// \brief Multiplication operator for Dimension.
        /// \param dim Right operand for multiplicaiton.
        /// \return 0 if either of `*this` or `dim` is static and 0; else, Dimension::dynamic() if
        ///         either of `*this` or `dim` is dynamic; else, a static dimension with value
        ///         `int64_t(*this)*int64_t(dim)`.
        Dimension operator*(const Dimension& dim) const;

        /// \brief Add-into operator for Dimension.
        /// \param dim Right operand for addition.
        /// \return A reference to `*this`, after updating `*this` to the value `*this + dim`.
        Dimension& operator+=(const Dimension& dim) { return (*this = *this + dim); }
        /// \brief Multiply-into operator for Dimension.
        /// \param dim Right operand for multiplication.
        /// \return A reference to `*this`, after updating `*this` to the value `*this * dim`.
        Dimension& operator*=(const Dimension& dim) { return (*this = *this * dim); }
    private:
        // The actual numerical value of the dimension. s_dynamic_val is a special case,
        // representing a dynamic dimension.
        int64_t m_dimension;
    };

    /// \brief Insert a human-readable representation of a dimension into an output stream.
    /// \param str The output stream targeted for insertion.
    /// \param dimension The dimension to be inserted into `str`.
    /// \return A reference to `str` after insertion.
    ///
    /// Inserts the string `?` if `dimension` is dynamic; else inserts `int64_t(dimension)`.
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);
}
