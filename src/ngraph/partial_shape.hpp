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

#include <stddef.h>

#include "ngraph/dimension.hpp"
#include "ngraph/rank.hpp"

namespace ngraph
{
    class PartialShape
    {
    public:
        PartialShape(std::initializer_list<Dimension> init)
            : PartialShape(true, init)
        {
        }
        bool rank_is_determined() const { return m_rank_is_determined; }
        bool is_complete() const;
        Rank rank() const
        {
            return m_rank_is_determined ? Rank(m_dimensions.size()) : Rank::undetermined();
        }
        friend std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
        friend PartialShape operator+(const PartialShape& s1, const PartialShape& s2);
        PartialShape append(const PartialShape& other);
        static PartialShape undetermined() { return PartialShape(false, {}); }
    private:
        PartialShape(bool rank_is_determined, std::initializer_list<Dimension> init)
            : m_rank_is_determined(rank_is_determined)
            , m_dimensions(init)
        {
        }
        bool m_rank_is_determined;
        std::vector<Dimension> m_dimensions;
    };

    PartialShape operator+(const PartialShape& s1, const PartialShape& s2);
    std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
}
