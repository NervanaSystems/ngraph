/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace ngraph
{
    /// \brief A difference (signed) of tensor element coordinates.
    class CoordinateDiff : public std::vector<std::ptrdiff_t>
    {
    public:
        CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        CoordinateDiff(const CoordinateDiff& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        explicit CoordinateDiff(size_t n, std::ptrdiff_t initial_value = 0)
            : std::vector<std::ptrdiff_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        CoordinateDiff(InputIterator first, InputIterator last)
            : std::vector<std::ptrdiff_t>(first, last)
        {
        }

        CoordinateDiff() {}
        CoordinateDiff& operator=(const CoordinateDiff& v)
        {
            static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
            return *this;
        }
        CoordinateDiff& operator=(CoordinateDiff&& v)
        {
            static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff);
}
