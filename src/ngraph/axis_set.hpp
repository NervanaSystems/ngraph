// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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
#include <set>

namespace ngraph
{
    /// \brief A set of axes.
    class AxisSet : public std::set<size_t>
    {
    public:
        AxisSet(const std::initializer_list<size_t>& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet(const std::set<size_t>& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet(const AxisSet& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet() {}
        AxisSet& operator=(const AxisSet& v)
        {
            static_cast<std::set<size_t>*>(this)->operator=(v);
            return *this;
        }
        AxisSet& operator=(AxisSet&& v)
        {
            static_cast<std::set<size_t>*>(this)->operator=(v);
            return *this;
        }
    };
}
