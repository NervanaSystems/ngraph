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

#include <algorithm>
#include <iostream>
#include <vector>

#include "ngraph/partial_shape.hpp"

using namespace ngraph;

PartialShape::PartialShape(const Shape& shape)
    : PartialShape(true, {})
{
    m_dimensions.assign(shape.begin(), shape.end());
}

bool ngraph::PartialShape::is_complete() const
{
    return m_rank_is_determined &&
           std::all_of(m_dimensions.begin(), m_dimensions.end(), [](const Dimension& d) {
               return d.is_determined();
           });
}

PartialShape ngraph::operator+(const PartialShape& s1, const PartialShape& s2)
{
    if (!s1.rank_is_determined() || !s2.rank_is_determined())
    {
        return PartialShape::undetermined();
    }

    if (!s1.rank().compatible(s2.rank()))
    {
        throw std::invalid_argument("rank mismatch");
    }

    PartialShape result{};
    result.m_rank_is_determined = true;
    for (size_t i = 0; i < s1.m_dimensions.size(); i++)
    {
        result.m_dimensions.push_back(s1.m_dimensions[i] + s2.m_dimensions[i]);
    }
    return result;
}

std::ostream& ngraph::operator<<(std::ostream& str, const PartialShape& shape)
{
    if (shape.m_rank_is_determined)
    {
        str << "{";
        bool first = true;
        for (auto& d : shape.m_dimensions)
        {
            if (!first)
            {
                str << ",";
            }
            str << d;
            first = false;
        }
        return (str << "}");
    }
    else
    {
        return (str << "?");
    }
}

bool PartialShape::compatible(const PartialShape& s) const
{
    // If we don't know *this's rank, or we don't know s's rank, they are compatible.
    if (!rank_is_determined() || !s.rank_is_determined())
    {
        return true;
    }
    // If we do know *this's rank and s's rank, and they are unequal, they are incompatible.
    else if (size_t(rank()) != size_t(s.rank()))
    {
        return false;
    }
    // If we know both the ranks and they are equal, then *this and s are compatible iff they
    // are elementwise compatible everywhere.
    else
    {
        for (size_t i = 0; i < size_t(rank()); i++)
        {
            if (!m_dimensions[i].compatible(s.m_dimensions[i]))
            {
                return false;
            }
        }
        // If we are still here, we know that s1 and s2 have the same rank and are elementwise
        // compatible everywhere.
        return true;
    }
}

bool PartialShape::same_scheme(const PartialShape& s) const
{
    if (!rank().is_determined() && !s.rank().is_determined())
    {
        return true;
    }
    else if (rank().is_determined() && s.rank().is_determined())
    {
        if (size_t(rank()) != size_t(s.rank()))
        {
            return false;
        }

        bool success = true;

        for (size_t i = 0; i < size_t(rank()); i++)
        {
            success &= (*this)[i].same_scheme(s[i]);
        }

        return success;
    }
    else
    {
        return false;
    }
}

Shape PartialShape::to_shape() const
{
    if (!is_complete())
    {
        throw std::invalid_argument("to_shape was called on an incomplete shape.");
    }

    return Shape(m_dimensions.begin(), m_dimensions.end());
}

bool PartialShape::merge_into(PartialShape& dst, const PartialShape& src)
{
    if (!dst.rank().is_determined())
    {
        dst = src;
        return true;
    }
    else if (!src.rank().is_determined())
    {
        // No change to dst.
        return true;
    }
    else if (size_t(dst.rank()) != size_t(src.rank()))
    {
        // Mismatching and determined ranks, cannot merge.
        return false;
    }
    else
    {
        // Ranks are both determined and match.
        bool success = true;
        for (size_t i = 0; i < size_t(dst.rank()); i++)
        {
            success &= Dimension::merge(dst[i], dst[i], src[i]);
        }
        return success;
    }
}
