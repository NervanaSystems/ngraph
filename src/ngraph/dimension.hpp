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

// XXX: THIS CLASS IS NOT IN USE YET AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.

#pragma once

#include <limits>
#include <stddef.h>

namespace ngraph
{
    /// \brief Class representing a possibly-unknown dimension in a shape or shape-like object.
    ///
    ///        Known dimensions may be implicitly converted from size_t. An unknown dimension is
    ///        constructed with Dimension() or Dimension::undetermined().
    class Dimension
    {
    public:
        /// \brief Constructs a known dimension.
        Dimension(size_t dimension)
            : m_dimension(dimension)
        {
        }

        /// \brief Constructs an unknown dimension.
        Dimension()
            : m_dimension(s_undetermined_val)
        {
        }

        /// \brief Returns true if this dimension is known.
        bool is_determined() const { return m_dimension != s_undetermined_val; }
        /// \brief Converts this dimension to size_t. If the dimension is unknown, behavior is
        ///        undefined.
        explicit operator size_t() const { return m_dimension; }
        /// \brief Constructs an unknown dimension.
        static Dimension undetermined() { return s_undetermined_val; }
    private:
        // The actual numerical value of the dimension. s_undetermined_val is a special case,
        // representing an unknown dimension.
        size_t m_dimension;

        // Constant for the size_t value used to represent an unknown dimension.
        static const size_t s_undetermined_val{std::numeric_limits<size_t>::max()};
    };

    /// \brief Pushes a human-readable representation of "dimension" onto "str".
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);

    /// \brief Addition operator for dimensions.
    ///
    ///        If d1 and d2 are both known, returns size_t(d1)+size_t(d2). Otherwise, returns
    ///        Dimension::undetermined().
    Dimension operator+(const Dimension& d1, const Dimension& d2);

    /// \brief Equality operator for dimensions.
    ///
    ///        If d1 and d2 are both known, returns size_t(d1)==size_t(d2). Otherwise, returns
    ///        false.
    bool operator==(const Dimension& d1, const Dimension& d2);

    /// \brief Inequality operator for dimensions.
    ///
    ///        If d1 and d2 are both known, returns size_t(d1)!=size_t(d2). Otherwise, returns
    ///        false.
    bool operator!=(const Dimension& d1, const Dimension& d2);
}
