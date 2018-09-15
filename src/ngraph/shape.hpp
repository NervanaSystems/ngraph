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
    class Shape
    {
    public:
        using value_type = size_t;
        Shape(const std::initializer_list<size_t>& axis_lengths)
            : m_data(axis_lengths)
        {
        }

        Shape(const std::vector<size_t>& axis_lengths)
            : m_data(axis_lengths)
        {
        }

        Shape(const Shape& axis_lengths)
            : m_data(axis_lengths.m_data)
        {
        }

        explicit Shape(size_t n, size_t initial_value = 0)
            : m_data(n, initial_value)
        {
        }

        template <class InputIterator>
        Shape(InputIterator first, InputIterator last)
            : m_data(first, last)
        {
        }

        Shape() {}
        Shape& operator=(const Shape& v)
        {
            m_data = v.m_data;
            return *this;
        }

        size_t size() const { return m_data.size(); }
        operator std::vector<size_t>() const { return m_data; }
        bool operator==(const Shape& other) const { return m_data == other.m_data; }
        bool operator!=(const Shape& other) const { return m_data != other.m_data; }
        std::vector<size_t>::iterator begin() { return m_data.begin(); }
        std::vector<size_t>::reference back() { return m_data.back(); }
        std::vector<size_t>::iterator end() { return m_data.end(); }
        std::vector<size_t>::const_iterator begin() const { return m_data.begin(); }
        std::vector<size_t>::const_reference back() const { return m_data.back(); }
        std::vector<size_t>::const_iterator end() const { return m_data.end(); }
        std::vector<size_t>::reverse_iterator rbegin() { return m_data.rbegin(); }
        std::vector<size_t>::reverse_iterator rend() { return m_data.rend(); }
        std::vector<size_t>::const_reverse_iterator rbegin() const { return m_data.rbegin(); }
        std::vector<size_t>::const_reverse_iterator rend() const { return m_data.rend(); }
        const size_t& operator[](size_t index) const { return m_data[index]; }
        size_t& operator[](size_t index) { return m_data[index]; }
        const size_t& at(size_t index) const { return m_data.at(index); }
        size_t& at(size_t index) { return m_data.at(index); }
        std::vector<size_t>::iterator insert(std::vector<size_t>::iterator pos, size_t value)
        {
            return m_data.insert(pos, value);
        }
        template <class InputIterator>
        void insert(std::vector<size_t>::iterator pos, InputIterator first, InputIterator last)
        {
            m_data.insert(pos, first, last);
        }
        void push_back(size_t value) { m_data.push_back(value); }
        std::vector<size_t>::iterator erase(std::vector<size_t>::iterator pos)
        {
            return m_data.erase(pos);
        }
        bool empty() const { return m_data.empty(); }
        const std::vector<size_t>& get_value() const { return m_data; }
    private:
        std::vector<size_t> m_data;
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
