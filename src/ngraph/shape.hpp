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

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/gpu_shape.hpp"
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
        operator GPUShape() const
        {
            GPUShape shape;
            for (size_t const& size : *this)
            {
                uint32_t low = static_cast<uint32_t>(size);
                if (low != size)
                {
                    throw std::runtime_error(
                        "Request for Shape which exceeds the bitwidth available for GPUShapes "
                        "(32)");
                }
                shape.push_back(low);
            }
            return shape;
        }
    };

    /// Number of elements in spanned by a shape
    template <typename T>
    auto shape_size(const T& shape) -> typename T::value_type
    {
        size_t size = 1;
        for (auto d : shape)
        {
            size *= d;
        }
        return size;
    }

    /// Row-major strides for a shape
    template <typename T>
    std::vector<typename T::value_type> row_major_strides(const T& shape)
    {
        std::vector<typename T::value_type> strides(shape.size());
        typename T::value_type s = 1;
        for (auto d = shape.rbegin(), st = strides.rbegin(); d != shape.rend(); d++, st++)
        {
            *st = s;
            s *= *d;
        }
        return strides;
    }

    template <typename T>
    inline bool is_scalar(const T& shape)
    {
        return 0 == shape.size();
    }
    template <typename T>
    inline bool is_vector(const T& shape)
    {
        return 1 == shape.size();
    }
}
