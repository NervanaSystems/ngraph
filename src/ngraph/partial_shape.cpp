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
#include <vector>

#include "ngraph/partial_shape.hpp"

using namespace ngraph;

bool ngraph::PartialShape::is_complete() const
{
    return m_rank_is_determined && std::all_of(m_lengths.begin(),
                                               m_lengths.end(),
                                               [](const Length& l) { return l.is_determined(); });
}

ngraph::PartialShape ngraph::operator+(const PartialShape& s1, const PartialShape& s2)
{
    if (!s1.rank_is_determined() || !s2.rank_is_determined())
    {
        return undetermined;
    }

    if (s1.rank() != s2.rank())
    {
        throw std::invalid_argument("rank mismatch");
    }

    PartialShape result{};
    result.m_rank_is_determined = true;
    for (size_t i = 0; i < s1.m_lengths.size(); i++)
    {
        result.m_lengths.push_back(s1.m_lengths[i] + s2.m_lengths[i]);
    }
    return result;
}

std::ostream& ngraph::operator<<(std::ostream& str, const PartialShape& shape)
{
    if (shape.m_rank_is_determined)
    {
        str << "{";
        bool first = true;
        for (auto& l : shape.m_lengths)
        {
            if (!first)
            {
                str << ",";
            }
            str << l;
            first = false;
        }
        return (str << "}");
    }
    else
    {
        return (str << "?");
    }
}
