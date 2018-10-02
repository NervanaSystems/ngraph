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

#include <stddef.h>

#include "ngraph/dimension.hpp"
#include "ngraph/rank.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    /// \brief Class representing a shape that may only be partially known.
    ///
    ///        XXX: THIS CLASS IS EXPERIMENTAL AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    ///
    ///        A partially-known shape may have:
    ///
    ///        - Unknown rank.
    ///        - Known rank, but unknown dimensions on some or all axes.
    ///        - Known rank, and known dimensions on all axes.
    class PartialShape
    {
    public:
        /// \brief Constructs a shape with determined rank.
        ///
        ///        Examples:
        ///
        ///        PartialShape s{2,3,4};                          // rank=3, all dimensions determined
        ///        PartialShape s{};                               // rank=0
        ///        PartialShape s{2,Dimension::undetermined(),3};  // rank=2, dimension 1 undetermined
        PartialShape(std::initializer_list<Dimension> init)
            : PartialShape(true, init)
        {
        }

        /// \brief Constructs a PartialShape with determined rank from a vector of Dimensions.
        PartialShape(const std::vector<Dimension>& dimensions)
            : m_rank_is_determined(true)
            , m_dimensions(dimensions)
        {
        }

        /// \brief Constructs a complete PartialShape with zero rank.
        PartialShape()
            : PartialShape({})
        {
        }

        /// \brief Constructs a complete PartialShape from a Shape.
        PartialShape(const Shape& shape);

        /// \brief Returns true if the shape has determined rank.
        bool rank_is_determined() const { return m_rank_is_determined; }
        /// \brief Returns true if the shape has known rank and all dimensions of the shape
        ///        are determined.
        bool is_complete() const;

        /// \brief Returns the rank of the shape. Returns Rank::undetermined() if the rank is undetermined.
        Rank rank() const
        {
            return m_rank_is_determined ? Rank(m_dimensions.size()) : Rank::undetermined();
        }

        /// \brief Appends another shape to this shape.
        ///
        ///        If "this" and "other" both have determined rank, returns a new shape two shape
        ///        whose dimensions are the concatenation of the dimensions of "this" and "other".
        ///        If either "this" or "other" has undetermined rank, returns
        ///        PartialShape::undetermined().
        PartialShape append(const PartialShape& other);

        /// \brief Returns the undetermined shape.
        static PartialShape undetermined() { return PartialShape(false, {}); }
        /// \brief Returns true if *this is compatible with s.
        ///
        ///        Two shapes are compatible if one or both of them has undetermined rank, or if
        ///        they both have determined and equal rank, and their dimensions are elementwise
        ///        compatible (see Dimension::compatible()).
        bool compatible(const PartialShape& s) const;

        /// \brief Returns true if *this has the same scheme as s.
        ///
        ///        Two shapes have the same scheme if they both have undetermined rank, or if they
        ///        both have determined and equal rank, and their dimensions have the same scheme
        ///        elementwise (see Dimension::same_scheme()).
        bool same_scheme(const PartialShape& s) const;

        /// \brief Converts a complete PartialShape to a Shape.
        ///
        ///        Throws std::invalid_argument if the PartialShape is incomplete.
        Shape to_shape() const;

        /// \brief Index operator for PartialShape.
        const Dimension& operator[](size_t i) const { return m_dimensions[i]; }
        /// \brief Index operator for PartialShape.
        Dimension& operator[](size_t i) { return m_dimensions[i]; }
        friend std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
        friend PartialShape operator+(const PartialShape& s1, const PartialShape& s2);

        /// \brief Merges src into dst, returning true on success and false on failure. If false
        ///        is returned, the effect on dst is unspecified.
        ///
        ///        To "merge" two partial shapes s1 and s2 is to find the most permissive partial
        ///        shape s that is no more permissive than s1 or s2, if s exists. For example:
        ///
        ///        merge(?,?) -> ?
        ///        merge(?,{?,?}) -> {?,?}
        ///        merge({?,?},{?,?}) -> {?,?}
        ///        merge({1,2,3,4},?) -> {1,2,3,4}
        ///        merge({1,2},{1,?}) -> {1,2}
        ///        merge({1,2,?,?},{1,?,3,?}) -> {1,2,3,?}
        ///        merge({1,2,3},{1,2,3}) -> {1,2,3}
        ///
        ///        merge({1,?},{2,?}) fails [dimension 0 constraints are inconsistent]
        ///        merge({?,?},{?,?,?}) fails [ranks are inconsistent]
        ///
        ///        This function (merge_into) performs the "merge" operation described above on
        ///        dst and src, but overwrites dst with the result and returns true if merging is
        ///        successful; if merging is unsuccessful, the function returns false and may make
        ///        unspecified changes to dst.
        static bool merge_into(PartialShape& dst, const PartialShape& src);

    private:
        // Private constructor so PartialShape::undetermined() can construct an undetermined shape.
        PartialShape(bool rank_is_determined, std::initializer_list<Dimension> init)
            : m_rank_is_determined(rank_is_determined)
            , m_dimensions(init)
        {
        }

        // True if the shape's rank is determined.
        bool m_rank_is_determined;

        // Shape dimensions. This has no meaning if m_rank_is_determined is false.
        std::vector<Dimension> m_dimensions;
    };

    /// \brief Elementwise addition of two shapes.
    ///
    ///        If s1 or s2 has undetermined rank, returns PartialShape::undetermined().
    ///        If s1 and s2 both have determined rank, and their ranks are unequal,
    ///           throws std::invalid_argument.
    ///        If s1 and s2 both have determined rank, and their ranks are equal,
    ///           returns a new shape whose ith dimension is s1[i] + s2[i].
    PartialShape operator+(const PartialShape& s1, const PartialShape& s2);

    /// \brief Inserts a human-readable representation of "shape" into "str".
    std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
}
