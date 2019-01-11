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

#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            /// \brief ngraph::Shape for a tensor resident on NVIDIAGPU.
            class Shape : public std::vector<uint32_t>
            {
            public:
                Shape(const std::initializer_list<uint32_t>& axis_lengths)
                    : std::vector<uint32_t>(axis_lengths)
                {
                }

                Shape(const std::vector<uint32_t>& axis_lengths)
                    : std::vector<uint32_t>(axis_lengths)
                {
                }

                Shape(const Shape& axis_lengths)
                    : std::vector<uint32_t>(axis_lengths)
                {
                }

                explicit Shape(size_t n, uint32_t initial_value = 0)
                    : std::vector<uint32_t>(n, initial_value)
                {
                }

                template <class InputIterator>
                Shape(InputIterator first, InputIterator last)
                    : std::vector<uint32_t>(first, last)
                {
                }

                Shape() {}
                Shape& operator=(const Shape& v)
                {
                    static_cast<std::vector<uint32_t>*>(this)->operator=(v);
                    return *this;
                }

                Shape& operator=(Shape&& v)
                {
                    static_cast<std::vector<uint32_t>*>(this)->operator=(v);
                    return *this;
                }

                Shape(const std::vector<size_t>& vec)
                {
                    for (size_t const& size : vec)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request exceeds the bitwidth available for Shapes (32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }

                Shape(const ngraph::Shape& shape)
                {
                    for (size_t const& size : shape)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request for ngraph::Shape which exceeds the bitwidth available "
                                "for Shapes "
                                "(32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }

                Shape(const Strides& strides)
                {
                    for (size_t const& size : strides)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request for Strides which exceed the bitwidth available for "
                                "Shapes "
                                "(32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }

                Shape(const Coordinate& coord)
                {
                    for (size_t const& size : coord)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request for Coordinate which exceed the bitwidth available for "
                                "Shapes "
                                "(32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }

                Shape(const AxisVector& vec)
                {
                    for (auto const& size : vec)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request for axis vector which exceed the bitwidth available for "
                                "Shapes "
                                "(32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }

                Shape(const AxisSet& axes_set)
                {
                    for (auto const& size : axes_set)
                    {
                        if (size >> 32 != 0)
                        {
                            throw std::runtime_error(
                                "Request for axis set which exceed the bitwidth available for "
                                "Shapes "
                                "(32)");
                        }
                        this->push_back(static_cast<uint32_t>(size));
                    }
                }
            };
        } // namespace nvidiagpu
    }     // namespace runtime
} // namespace ngraph
