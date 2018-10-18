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

#pragma once

#include <algorithm>
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    /// \brief Coordinates for a tensor element
    class Coordinate : public std::vector<size_t>
    {
    public:
        Coordinate() {}
        Coordinate(const std::initializer_list<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Shape& shape)
            : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape))
        {
        }

        Coordinate(const std::vector<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Coordinate& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Coordinate(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Coordinate& operator=(const Coordinate& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }

        Coordinate& operator=(Coordinate&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);
}
