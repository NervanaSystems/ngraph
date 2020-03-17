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

#include <algorithm>
#include <iostream>
#include <vector>

#include "ngraph/check.hpp"
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
        r.is_static(),
        std::vector<Dimension>(r.is_static() ? r.get_length() : 0, Dimension::dynamic()));
}

bool PartialShape::compatible(const PartialShape& s) const
{
    // If we don't know *this's rank, or we don't know s's rank, they are compatible.
    if (!m_rank_is_static || s.rank().is_dynamic())
    {
        return true;
    }
    // If we do know *this's rank and s's rank, and they are unequal, they are incompatible.
    else if (rank().get_length() != s.rank().get_length())
    {
        return false;
    }
    // If we know both the ranks and they are equal, then *this and s are compatible iff they
    // are elementwise compatible everywhere.
    else
    {
        for (size_t i = 0; i < rank().get_length(); i++)
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
        if (rank().get_length() != s.rank().get_length())
        {
            return false;
        }

        bool success = true;

        for (size_t i = 0; i < rank().get_length(); i++)
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
    else if (s.rank().is_static() && rank().get_length() == s.rank().get_length())
    {
        bool all_relax = true;

        for (size_t i = 0; i < rank().get_length(); i++)
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
    else if (rank().is_static() && rank().get_length() == s.rank().get_length())
    {
        bool all_refine = true;

        for (size_t i = 0; i < rank().get_length(); i++)
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
        m_dimensions = std::vector<Dimension>(r.get_length(), Dimension::dynamic());
        return true;
    }
    else
    {
        return (m_dimensions.size() == r.get_length());
    }
}

Shape PartialShape::to_shape() const
{
    if (is_dynamic())
    {
        throw std::invalid_argument("to_shape was called on a dynamic shape.");
    }

    std::vector<size_t> dimensions_to_shape(m_dimensions.size());
    std::transform(m_dimensions.begin(),
                   m_dimensions.end(),
                   dimensions_to_shape.begin(),
                   [](const Dimension& d) { return d.get_length(); });

    return Shape(dimensions_to_shape.begin(), dimensions_to_shape.end());
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
    else if (dst.rank().get_length() != src.rank().get_length())
    {
        // Mismatching static ranks, cannot merge.
        return false;
    }
    else
    {
        // Ranks are both static, and they match.
        bool success = true;
        for (size_t i = 0; i < dst.rank().get_length(); i++)
        {
            success &= Dimension::merge(dst[i], dst[i], src[i]);
        }
        return success;
    }
}

bool PartialShape::broadcast_merge_into(PartialShape& dst,
                                        const PartialShape& src,
                                        const op::AutoBroadcastSpec& autob)
{
    switch (autob.m_type)
    {
    case op::AutoBroadcastType::NONE: return true;
    case op::AutoBroadcastType::NUMPY:
    {
        if (dst.rank().is_dynamic() || src.rank().is_dynamic())
        {
            dst = PartialShape::dynamic();
            return true;
        }
        else
        {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            auto new_rank = std::max(dst_rank, src_rank);
            std::vector<Dimension> dims(new_rank);
            bool success = true;
            for (size_t i = 0; i < new_rank; i++)
            {
                auto dsti =
                    i < (new_rank - dst_rank) ? Dimension(1) : dst[i - (new_rank - dst_rank)];
                auto srci =
                    i < (new_rank - src_rank) ? Dimension(1) : src[i - (new_rank - src_rank)];
                success &= Dimension::broadcast_merge(dims[i], dsti, srci);
            }
            dst = PartialShape(dims);
            return success;
        }
    }
    case op::AutoBroadcastType::PDPD:
    {
        if (dst.rank().is_dynamic() || src.rank().is_dynamic())
        {
            return true;
        }
        else
        {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            if (dst_rank == src_rank && dst.compatible(src))
                return true;

            int64_t axis = autob.m_axis;
            if (axis < -1)
            {
                return false;
            }
            if (axis == -1)
            {
                axis = dst_rank - src_rank;
            }

            size_t len = src_rank;
            while (len > 0 && src[len - 1].is_static() && src[len - 1].get_length() == 1)
            {
                --len;
            }

            for (size_t i = axis; i < axis + len; ++i)
            {
                if (!(dst[i].compatible(src[i - axis])))
                {
                    return false;
                }
            }

            return true;
        }
    }
    default: NGRAPH_CHECK(false, "Unsupported auto broadcast type: ", autob.m_type);
    }

    return false;
}

bool PartialShape::all_non_negative() const
{
    for (auto& d : m_dimensions)
    {
        if (d.is_static() && int64_t(d) < 0)
        {
            return false;
        }
    }

    return true;
}

NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<PartialShape>::type_info;
