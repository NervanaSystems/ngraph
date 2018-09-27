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

#include "ngraph/dimension.hpp"

using namespace ngraph;

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

bool Dimension::compatible(const Dimension& d) const
{
    return (!is_determined() || !d.is_determined() || m_dimension == size_t(d));
}
