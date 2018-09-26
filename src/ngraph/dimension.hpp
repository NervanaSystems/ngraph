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

// XXX: THIS CLASS IS NOT IN USE YET AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.

#pragma once

#include <limits>
#include <stddef.h>

namespace ngraph
{
    class Dimension
    {
    public:
        Dimension(size_t dimension)
            : m_dimension(dimension)
        {
        }
        Dimension()
            : m_dimension(s_undetermined_val)
        {
        }
        bool is_determined() const { return m_dimension != s_undetermined_val; }
        explicit operator size_t() const { return m_dimension; }
        static const Dimension& undetermined() { return s_undetermined; }
    private:
        size_t m_dimension;
        static const Dimension& s_undetermined;
        static const size_t s_undetermined_val{std::numeric_limits<size_t>::max()};
    };

    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);
    Dimension operator+(const Dimension& d1, const Dimension& d2);
    bool operator==(const Dimension& d1, const Dimension& d2);
    bool operator!=(const Dimension& d1, const Dimension& d2);
}
