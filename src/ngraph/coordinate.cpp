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

#include "ngraph/coordinate.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Coordinate& coordinate)
{
    s << "Coordinate{";
    s << ngraph::join(coordinate.m_data);
    s << "}";
    return s;
}

ngraph::Coordinate::Coordinate() {}

ngraph::Coordinate::Coordinate(const std::initializer_list<size_t>& axes)
    : m_data{axes}
{
}

ngraph::Coordinate::Coordinate(const Shape& shape)
    : m_data()
{
    for (auto s : shape)
    {
        m_data.push_back(s);
    }
}

ngraph::Coordinate::Coordinate(const std::vector<size_t>& axes)
    : m_data(axes)
{
}

ngraph::Coordinate::Coordinate(const Coordinate& axes)
    : m_data(axes.m_data)
{
}

ngraph::Coordinate::Coordinate(size_t n, size_t initial_value)
    : m_data(n, initial_value)
{
}

ngraph::Coordinate::~Coordinate() {}

ngraph::Coordinate& ngraph::Coordinate::operator=(const Coordinate& v)
{
    m_data = v.m_data;
    return *this;
}

ngraph::Coordinate& ngraph::Coordinate::operator=(Coordinate&& v) noexcept
{
    m_data = v.m_data;
    return *this;
}

constexpr ngraph::DiscreteTypeInfo ngraph::AttributeAdapter<ngraph::Coordinate>::type_info;
