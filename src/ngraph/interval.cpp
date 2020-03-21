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

#include "ngraph/interval.hpp"

using namespace ngraph;

Interval::Interval(value_type min_val, value_type max_val)
    : m_min_val(clip(min_val))
    , m_max_val(clip(max_val))
{
}

Interval::Interval(value_type val, Extent extent)
    : Interval(extent == Extent::Below ? s_min : val, extent == Extent::Above ? s_max : val)
{
}

Interval::size_type Interval::size() const
{
    return m_min_val <= m_max_val ? m_max_val - m_min_val + 1 : 0;
}

bool Interval::empty() const
{
    return m_min_val > m_max_val;
}

bool Interval::operator==(const Interval& interval) const
{
    return m_min_val == interval.m_min_val && m_max_val == interval.m_max_val;
}

bool Interval::operator!=(const Interval& interval) const
{
    return m_min_val != interval.m_min_val || m_max_val != interval.m_max_val;
}

Interval Interval::operator+(const Interval& interval) const
{
    return Interval(m_min_val + interval.m_min_val, m_max_val + interval.m_max_val);
}

Interval& Interval::operator+=(const Interval& interval)
{
    m_min_val += interval.m_min_val;
    m_max_val += interval.m_max_val;
    return *this;
}

Interval Interval::operator-(const Interval& interval) const
{
    return Interval(m_min_val - interval.m_max_val, m_max_val - interval.m_min_val);
}

Interval& Interval::operator-=(const Interval& interval)
{
    m_min_val -= interval.m_max_val;
    m_max_val -= interval.m_min_val;
    return *this;
}

Interval Interval::operator*(const Interval& interval) const
{
    return Interval(clip_times(m_min_val, interval.m_min_val),
                    clip_times(m_max_val, interval.m_max_val));
}

Interval& Interval::operator*=(const Interval& interval)
{
    m_min_val = clip_times(m_min_val, interval.m_min_val);
    m_max_val = clip_times(m_max_val, interval.m_max_val);
    return *this;
}

Interval Interval::operator&(const Interval& interval) const
{
    return Interval(std::max(m_min_val, interval.m_min_val),
                    std::min(m_max_val, interval.m_max_val));
}

Interval& Interval::operator&=(const Interval& interval)
{
    m_min_val = std::max(m_min_val, interval.m_min_val);
    m_max_val = std::min(m_max_val, interval.m_max_val);
    return *this;
}

bool Interval::contains(value_type value) const
{
    return m_min_val <= value && value <= m_max_val;
}

bool Interval::contains(const Interval& interval) const
{
    return contains(interval.m_min_val) && contains(interval.m_max_val);
}

Interval::value_type Interval::clip(value_type value)
{
    return std::max(s_min, std::min(s_max, value));
}

Interval::value_type Interval::clip_times(value_type a, value_type b)
{
    if (a == 0 || b == 0)
    {
        return 0;
    }
    else if (a == s_max)
    {
        return b < 0 ? s_min : s_max;
    }
    else if (a == s_min)
    {
        return b < 0 ? s_max : s_min;
    }
    else if (b == s_max)
    {
        return a < 0 ? s_min : s_max;
    }
    else if (b == s_min)
    {
        return a < 0 ? s_max : s_min;
    }
    else
        return a * b;
}

Interval::value_type Interval::s_min;
Interval::value_type Interval::s_max;

namespace ngraph
{
    std::ostream& operator<<(std::ostream& str, const Interval& interval)
    {
        return str << "Interval(" << interval.get_min_val() << ", " << interval.get_max_val()
                   << ")";
    }
}
