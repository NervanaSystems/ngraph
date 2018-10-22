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

bool ngraph::PartialShape::is_static() const
{
    return m_rank_is_static && std::all_of(m_dimensions.begin(),
                                           m_dimensions.end(),
                                           [](const Dimension& d) { return d.is_static(); });
}

PartialShape ngraph::operator+(const PartialShape& s1, const PartialShape& s2)
{
    if (s1.rank().is_dynamic() || s2.rank().is_dynamic())
    {
        return PartialShape::dynamic();
    }

    if (!s1.rank().compatible(s2.rank()))
    {
        throw std::invalid_argument("rank mismatch");
    }

    PartialShape result{};
    result.m_rank_is_static = true;
    for (size_t i = 0; i < s1.m_dimensions.size(); i++)
    {
        result.m_dimensions.push_back(s1.m_dimensions[i] + s2.m_dimensions[i]);
    }
    return result;
}

std::ostream& ngraph::operator<<(std::ostream& str, const PartialShape& shape)
{
    if (shape.m_rank_is_static)
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

PartialShape PartialShape::dynamic(Rank r)
{
    return PartialShape(
        r.is_static(), std::vector<Dimension>(r.is_static() ? size_t(r) : 0, Dimension::dynamic()));
}

bool PartialShape::compatible(const PartialShape& s) const
{
    // If we don't know *this's rank, or we don't know s's rank, they are compatible.
    if (!m_rank_is_static || s.rank().is_dynamic())
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
    if (rank().is_dynamic() && s.rank().is_dynamic())
    {
        return true;
    }
    else if (rank().is_static() && s.rank().is_static())
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

bool PartialShape::relaxes(const PartialShape& s) const
{
    if (rank().is_dynamic())
    {
        return true;
    }
    else if (s.rank().is_static() && size_t(rank()) == size_t(s.rank()))
    {
        bool all_relax = true;

        for (size_t i = 0; i < size_t(rank()); i++)
        {
            all_relax &= ((*this)[i].relaxes(s[i]));
        }

        return all_relax;
    }
    else
    {
        return false;
    }
}

bool PartialShape::refines(const PartialShape& s) const
{
    if (s.rank().is_dynamic())
    {
        return true;
    }
    else if (rank().is_static() && size_t(rank()) == size_t(s.rank()))
    {
        bool all_refine = true;

        for (size_t i = 0; i < size_t(rank()); i++)
        {
            all_refine &= ((*this)[i].refines(s[i]));
        }

        return all_refine;
    }
    else
    {
        return false;
    }
}

bool PartialShape::merge_rank(Rank r)
{
    if (r.is_dynamic())
    {
        return true;
    }
    else if (!m_rank_is_static)
    {
        m_rank_is_static = true;
        m_dimensions = std::vector<Dimension>(size_t(r), Dimension::dynamic());
        return true;
    }
    else
    {
        return (m_dimensions.size() == size_t(r));
    }
}

Shape PartialShape::to_shape() const
{
    if (is_dynamic())
    {
        throw std::invalid_argument("to_shape was called on a dynamic shape.");
    }

    return Shape(m_dimensions.begin(), m_dimensions.end());
}

bool PartialShape::merge_into(PartialShape& dst, const PartialShape& src)
{
    if (dst.rank().is_dynamic())
    {
        dst = src;
        return true;
    }
    else if (src.rank().is_dynamic())
    {
        // No change to dst.
        return true;
    }
    else if (size_t(dst.rank()) != size_t(src.rank()))
    {
        // Mismatching static ranks, cannot merge.
        return false;
    }
    else
    {
        // Ranks are both static, and they match.
        bool success = true;
        for (size_t i = 0; i < size_t(dst.rank()); i++)
        {
            success &= Dimension::merge(dst[i], dst[i], src[i]);
        }
        return success;
    }
}
