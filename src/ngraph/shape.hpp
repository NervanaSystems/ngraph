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
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    /// \brief Shape for a tensor.
    class Shape : public std::vector<size_t>
    {
    public:
        Shape(const std::initializer_list<size_t>& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        Shape(const std::vector<size_t>& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        Shape(const Shape& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        explicit Shape(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Shape(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Shape() {}
        Shape& operator=(const Shape& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
        Shape& operator=(Shape&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    /// Number of elements in spanned by a shape
    template <typename SHAPE_TYPE>
    size_t shape_size(const SHAPE_TYPE& shape)
    {
        size_t size = 1;
        for (auto d : shape)
        {
            size *= d;
        }
        return size;
    }

    /// Row-major strides for a shape
    template <typename SHAPE_TYPE>
    std::vector<size_t> row_major_strides(const SHAPE_TYPE& shape)
    {
        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        auto st = strides.rbegin();
        for (auto d = shape.rbegin(); d != shape.rend(); d++, st++)
        {
            *st = s;
            s *= *d;
        }
        return strides;
    }

    template <typename SHAPE_TYPE>
    inline bool is_scalar(const SHAPE_TYPE& shape)
    {
        return 0 == shape.size();
    }

    template <typename SHAPE_TYPE>
    inline bool is_vector(const SHAPE_TYPE& shape)
    {
        return 1 == shape.size();
    }

    std::ostream& operator<<(std::ostream& s, const Shape& shape);
}
