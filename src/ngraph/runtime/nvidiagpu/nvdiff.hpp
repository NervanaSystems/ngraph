//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            /// \brief ngraph::CoordinateDiff for a tensor resident on NVIDIAGPU.
            class CoordinateDiff : public std::vector<int32_t>
            {
            public:
                CoordinateDiff(const std::initializer_list<int32_t>& axis_lengths)
                    : std::vector<int32_t>(axis_lengths)
                    {
                    }

                CoordinateDiff(const std::vector<int32_t>& axis_lengths)
                    : std::vector<int32_t>(axis_lengths)
                    {
                    }

                CoordinateDiff(const CoordinateDiff& axis_lengths)
                    : std::vector<int32_t>(axis_lengths)
                    {
                    }

                explicit CoordinateDiff(size_t n, int32_t initial_value = 0)
                    : std::vector<int32_t>(n, initial_value)
                    {
                    }

                template <class InputIterator>
                CoordinateDiff(InputIterator first, InputIterator last)
                    : std::vector<int32_t>(first, last)
                    {
                    }

                CoordinateDiff() {}
                CoordinateDiff& operator=(const CoordinateDiff& v)
                    {
                        static_cast<std::vector<int32_t>*>(this)->operator=(v);
                        return *this;
                    }

                CoordinateDiff& operator=(CoordinateDiff&& v)
                    {
                        static_cast<std::vector<int32_t>*>(this)->operator=(v);
                        return *this;
                    }

                CoordinateDiff(const ngraph::CoordinateDiff& coord)
                    {
                        for (auto const& dim : coord)
                        {
                            if (std::abs(dim) >> 32 != 0)
                            {
                                throw std::runtime_error(
                                    "Request for ngraph::CoordinateDiff which exceed the bitwidth available for "
                                    "CoordinateDiffs (32)");
                            }
                            this->push_back(static_cast<int32_t>(dim));
                        }
                    }
            };
        }
    }
}
