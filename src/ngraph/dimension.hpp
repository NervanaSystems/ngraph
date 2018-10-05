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
    /// \brief Class representing a possibly-unknown dimension in a shape or shape-like object.
    ///
    ///        Known dimensions may be implicitly converted from size_t. An unknown dimension is
    ///        constructed with Dimension() or Dimension::undetermined().
    ///
    /// XXX: THIS CLASS IS NOT IN USE YET AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    class Dimension
    {
    public:
        /// \brief Construct a determined dimension.
        /// \param dimension Value of the dimension. Must not be equal to
        ///                  Dimension::s_undetermined_val.
        /// \throws std::invalid_argument If `dimension` == Dimension::s_undetermined_val.
        Dimension(size_t dimension);

        /// \brief Construct an unknown dimension.
        Dimension() { m_dimension = s_undetermined_val; }
        /// \brief Check whether this dimension is determined.
        /// \return `true` if the dimension is determined, else `false`.
        bool is_determined() const { return m_dimension != s_undetermined_val; }
        /// \brief Convert this dimension to `size_t`. This dimension must not be unknown.
        /// \throws std::invalid_argument If this dimension is undetermined.
        explicit operator size_t() const
        {
            if (!is_determined())
            {
                throw std::invalid_argument("Cannot convert unknown dimension to size_t");
            }
            return m_dimension;
        }

        /// \brief Check whether this dimension represents the same scheme as the argument (both
        ///        undetermined, or equal).
        /// \param dim The other dimension to compare this dimension to.
        /// \return `true` if this dimension and `dim` are both undetermined, or if they are both
        ///         determined and equal; otherwise, `false`.
        bool same_scheme(const Dimension& dim) const
        {
            return (!is_determined() && !dim.is_determined()) ||
                   (is_determined() && dim.is_determined() && m_dimension == size_t(dim));
        }

        /// \brief Try to merge two Dimension objects together.
        /// \param[out] dst Reference to write the merged Dimension into.
        /// \param d1 First dimension to merge.
        /// \param d2 Second dimension to merge.
        /// \return `true` if merging succeeds, else `false`.
        ///
        /// \li If `d1` is undetermined, writes `d2` to `dst` and returns `true`.
        /// \li If `d2` is undetermined, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are determined and equal, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are both determined and unequal, leaves `dst` unchanged and
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
        /// \brief Create an unknown dimension.
        /// \return An unknown dimension.
        static Dimension undetermined() { return Dimension(); }
        /// \brief Constant for the value used internally to represent an unknown dimension.
        static const size_t s_undetermined_val{std::numeric_limits<size_t>::max()};

    private:
        // The actual numerical value of the dimension. s_undetermined_val is a special case,
        // representing an unknown dimension.
        size_t m_dimension;
    };

    /// \brief Insert a human-readable representation of a dimension into an output stream.
    /// \param str The output stream targeted for insertion.
    /// \param dimension The dimension to be inserted into `str`.
    /// \return A reference to `str` after insertion.
    ///
    /// Inserts the string `?` if `dimension` is undetermined; else inserts `size_t(dimension)`.
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);

    /// \brief Addition operator for dimensions.
    /// \param d1 Left operand for addition.
    /// \param d2 Right operand for addition.
    /// \return Dimension::undetermined() if `d1` or `d2` is unknown; else, a determined dimension
    ///         with value `size_t(d1)+size_t(d2)`.
    Dimension operator+(const Dimension& d1, const Dimension& d2);
}
