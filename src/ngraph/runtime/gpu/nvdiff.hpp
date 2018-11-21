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

#include <cstdio>
#include <stdexcept>
#include <vector>

#include "ngraph/coordinate_diff.hpp"

namespace ngraph
{
    class Shape;
    /// \brief Shape for a tensor resident on GPU.
    class NVDiff : public std::vector<int32_t>
    {
    public:
        NVDiff(const std::initializer_list<int32_t>& axis_lengths)
            : std::vector<int32_t>(axis_lengths)
        {
        }

        NVDiff(const std::vector<int32_t>& axis_lengths)
            : std::vector<int32_t>(axis_lengths)
        {
        }

        NVDiff(const NVDiff& axis_lengths)
            : std::vector<int32_t>(axis_lengths)
        {
        }

        explicit NVDiff(size_t n, int32_t initial_value = 0)
            : std::vector<int32_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        NVDiff(InputIterator first, InputIterator last)
            : std::vector<int32_t>(first, last)
        {
        }

        NVDiff() {}
        NVDiff& operator=(const NVDiff& v)
        {
            static_cast<std::vector<int32_t>*>(this)->operator=(v);
            return *this;
        }

        NVDiff& operator=(NVDiff&& v)
        {
            static_cast<std::vector<int32_t>*>(this)->operator=(v);
            return *this;
        }

        NVDiff(const CoordinateDiff& coord)
        {
            for (auto const& dim : coord)
            {
                if (std::abs(dim) >> 32 != 0)
                {
                    throw std::runtime_error(
                        "Request for CoordinateDiff which exceed the bitwidth available for "
                        "NVDiffs (32)");
                }
                this->push_back(static_cast<int32_t>(dim));
            }
        }
    };
}
