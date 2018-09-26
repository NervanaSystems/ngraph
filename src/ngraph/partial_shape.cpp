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

    if (s1.rank() != s2.rank())
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

bool ngraph::operator==(const PartialShape& s1, const PartialShape& s2)
{
    // If we don't know that the ranks are equal, we don't know that s1 and s2 are equal.
    if (s1.rank().possibly_neq(s2.rank()))
    {
        return false;
    }
    // If we do know that the ranks are equal, we check each component elementwise.
    else
    {
        for (size_t i = 0; i < size_t(s1.rank()); i++)
        {
            // If we don't know that these two corresponding elements are equal, we don't know
            // that s1 and s2 are equal.
            if (s1.m_dimensions[i].possibly_neq(s2.m_dimensions[i]))
            {
                return false;
            }
        }
        // If we are still here, we know that s1 and s2 have the same rank and are elementwise
        // necessarily equal everywhere.
        return true;
    }
}

bool ngraph::operator!=(const PartialShape& s1, const PartialShape& s2)
{
    // If we know that the ranks are unequal, we know s1 and s2 are unequal.
    if (s1.rank() != s2.rank())
    {
        return true;
    }
    // If we do not know that the ranks are unequal, and we do not know that they are equal,
    // then one of s1 or s2 has undetermined rank, and we do not know that s1 and s2 are unequal.
    else if (s1.rank().possibly_neq(s2.rank()))
    {
        return false;
    }
    // If we do know that the ranks are equal, we check each component elementwise.
    else
    {
        for (size_t i = 0; i < size_t(s1.rank()); i++)
        {
            // If we know that these two corresponding elemenats are not equal, we know that s1
            // and s2 are not equal.
            if (s1.m_dimensions[i] != s2.m_dimensions[i])
            {
                return true;
            }
        }
        // If we are still here, then we know that s1 and s2 have the same rank, but there is
        // nowhere that we know that s1 and s2 are elementwise unequal. Therefore we do not know
        // that s1 and s2 are unequal.
        return false;
    }
}
