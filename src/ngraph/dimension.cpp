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

#include <iostream>
#include <limits>
#include <sstream>

#include "ngraph/dimension.hpp"

using namespace ngraph;

Dimension::Dimension(size_t dimension)
    : m_dimension(dimension)
{
    if (dimension == s_undetermined_val)
    {
        std::stringstream ss;
        ss << "Cannot convert the value 0x" << std::uppercase << std::hex << s_undetermined_val
           << " to Dimension: this value is used internally to represent an undetermined "
              "dimension.";
        throw std::invalid_argument(ss.str());
    }
}

std::ostream& ngraph::operator<<(std::ostream& str, const Dimension& dimension)
{
    if (dimension.is_determined())
    {
        return (str << size_t(dimension));
    }
    else
    {
        return (str << "?");
    }
}

Dimension ngraph::operator+(const Dimension& d1, const Dimension& d2)
{
    return (d1.is_determined() && d2.is_determined() ? size_t(d1) + size_t(d2)
                                                     : Dimension::undetermined());
}

Dimension ngraph::operator*(const Dimension& d1, const Dimension& d2)
{
    return ((d1.is_determined() && d2.is_determined())
                ? size_t(d1) * size_t(d2)
                : (d1.is_determined() && size_t(d1) == 0)
                      ? 0
                      : (d2.is_determined() && size_t(d2) == 0) ? 0 : Dimension::undetermined());
}

bool Dimension::compatible(const Dimension& d) const
{
    return (!is_determined() || !d.is_determined() || m_dimension == size_t(d));
}

bool Dimension::merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    if (!d1.is_determined())
    {
        dst = d2;
        return true;
    }
    else if (!d2.is_determined())
    {
        dst = d1;
        return true;
    }
    else if (size_t(d1) != size_t(d2))
    {
        return false;
    }
    else
    {
        dst = d1;
        return true;
    }
}
