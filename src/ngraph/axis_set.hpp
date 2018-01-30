// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>

#include "ngraph/set_forwarder.hpp"

namespace ngraph
{
    /// \brief A set of axes.
    class AxisSet : public SetForwarder<size_t, AxisSet>
    {
    public:
        AxisSet(const std::initializer_list<size_t>& axes)
            : SetForwarder<size_t, AxisSet>(axes)
        {
        }

        AxisSet(const std::set<size_t>& axes)
            : SetForwarder<size_t, AxisSet>(axes)
        {
        }

        AxisSet(const AxisSet& axes)
            : SetForwarder<size_t, AxisSet>(axes)
        {
        }

        AxisSet() {}
        AxisSet& operator=(const AxisSet& v)
        {
            m_set = v.m_set;
            return *this;
        }
        AxisSet& operator=(AxisSet&& v)
        {
            m_set = v.m_set;
            return *this;
        }
    };
}
