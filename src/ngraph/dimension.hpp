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
        /// \brief Constructs a known dimension.
        ///
        ///        Requires that dimension != s_undetermined_val. If that condition does not hold,
        ///        throws std::invalid_argument.
        Dimension(size_t dimension);

        /// \brief Constructs an unknown dimension.
        Dimension() { m_dimension = s_undetermined_val; }
        /// \brief Returns true if this dimension is determined.
        bool is_determined() const { return m_dimension != s_undetermined_val; }
        /// \brief Converts this dimension to size_t. If the dimension is undetermined, throws
        ///        std::invalid_argument.
        explicit operator size_t() const
        {
            if (!is_determined())
            {
                throw std::invalid_argument("Cannot convert unknown dimension to size_t");
            }
            return m_dimension;
        }

        bool same_scheme(const Dimension& dim) const
        {
            return (!is_determined() && !dim.is_determined()) ||
                   (is_determined() && dim.is_determined() && m_dimension == size_t(dim));
        }

        /// \brief Merge two dimensions together. Returns true, and writes the merged dimension
        ///        to dst, if the merge is successful. Returns false, and leaves dst unchanged,
        ///        if the merge is unsuccessful.
        static bool merge(Dimension& dst, const Dimension d1, const Dimension d2);

        /// \brief Returns true if the dimensions are compatible, i.e. if one of the dimensions
        ///        is undetermined, or both dimensions are determined and equal.
        bool compatible(const Dimension& d) const;
        /// \brief Constructs an unknown dimension.
        static Dimension undetermined() { return Dimension(); }
        /// \brief Constant for the value used internally to represent an unknown dimension.
        static const size_t s_undetermined_val{std::numeric_limits<size_t>::max()};

    private:
        // The actual numerical value of the dimension. s_undetermined_val is a special case,
        // representing an unknown dimension.
        size_t m_dimension;
    };

    /// \brief Inserts a human-readable representation of "dimension" into "str".
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);

    /// \brief Addition operator for dimensions.
    ///
    ///        If d1 and d2 are both known, returns size_t(d1)+size_t(d2). Otherwise, returns
    ///        Dimension::undetermined().
    Dimension operator+(const Dimension& d1, const Dimension& d2);
}
