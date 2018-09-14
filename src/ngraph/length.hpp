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

#include "ngraph/undetermined.hpp"

namespace ngraph
{
    class Length
    {
    public:
        Length(size_t length)
            : m_length(length)
            , m_fixed(true)
        {
        }
        Length(const Undetermined&)
            : m_length(0)
            , m_fixed(false)
        {
        }
        Length()
            : m_length(0)
            , m_fixed(true)
        {
        }
        bool fixed() const { return m_fixed; }
        explicit operator size_t() const { return m_length; }
    private:
        size_t m_length;
        bool m_fixed;
    };

    std::ostream& operator<<(std::ostream& str, const Length& length);
    Length operator+(const Length& l1, const Length& l2);
}
