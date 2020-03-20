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
#include <limits>
#include <sstream>

#include "ngraph/dimension.hpp"

using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& str, const Dimension& dimension)
{
    if (dimension.is_static())
    {
        return str << dimension.get_length();
    }
    else
    {
        return str << "?";
    }
}

Dimension Dimension::operator+(const Dimension& dim) const
{
    return Dimension(m_dimension + dim.m_dimension);
}

Dimension Dimension::operator-(const Dimension& dim) const
{
    return Dimension(m_dimension - dim.m_dimension);
}

Dimension Dimension::operator*(const Dimension& dim) const
{
    return Dimension(m_dimension * dim.m_dimension);
}

bool Dimension::compatible(const Dimension& d) const
{
    return !(m_dimension & d.m_dimension).empty();
}

bool Dimension::relaxes(const Dimension& d) const
{
    return m_dimension.contains(d.m_dimension);
}

bool Dimension::refines(const Dimension& d) const
{
    return d.m_dimension.contains(m_dimension);
}

bool Dimension::same_scheme(const Dimension& dim) const
{
    return (m_dimension == dim.m_dimension) ||
           (m_dimension.size() > 1 && dim.m_dimension.size() > 1);
}

bool Dimension::merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    auto result = d1.m_dimension & d2.m_dimension;
    if (result.empty())
    {
        return false;
    }
    dst = result;
    return true;
}

bool Dimension::broadcast_merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    if (d1.m_dimension.size() == 1 && d1.m_dimension.get_min_val() == 1)
    {
        dst = d2;
        return true;
    }
    if (d2.m_dimension.size() == 1 && d2.m_dimension.get_min_val() == 1)
    {
        dst = d1;
        return true;
    }
    return merge(dst, d1, d2);
}

uint64_t Dimension::get_length() const
{
    if (is_dynamic())
    {
        throw std::invalid_argument("Cannot get length of dynamic dimension");
    }
    return m_dimension.get_min_val();
}

Dimension::operator size_t() const
{
    if (is_dynamic())
    {
        throw std::invalid_argument("Cannot convert dynamic dimension to size_t");
    }
    auto result = m_dimension.get_min_val();
    if (result > std::numeric_limits<size_t>::max())
    {
        throw std::invalid_argument("Dimension to large for size_t");
    }
    return result;
}
