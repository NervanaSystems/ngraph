// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
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

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

#include "ngraph/common.hpp"

namespace ngraph
{
    class CoordinateTransform
    {
    public:
        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order,
                            const Shape& source_padding_below,
                            const Shape& source_padding_above);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner);

        CoordinateTransform(const Shape& source_shape);

        size_t index(const Coordinate& c) const;
        bool in_bounds(const Coordinate& c) const;
        bool in_padding(const Coordinate& c) const;
        Coordinate to_source_coordinate(const Coordinate& c) const;
        const Coordinate& get_target_shape() const;

        const Shape& get_source_shape() const { return m_source_shape; }
        const Coordinate& get_source_start_corner() const { return m_source_start_corner; }
        const Coordinate& get_source_end_corner() const { return m_source_end_corner; }
        const Strides& get_source_strides() const { return m_source_strides; }
        const AxisVector& get_source_axis_order() const { return m_source_axis_order; }
        class Iterator
        {
        public:
            Iterator(const Shape& target_shape, bool is_end = false);

            void operator++();
            Iterator operator++(int);
            void operator+=(size_t n);
            Coordinate operator*();
            bool operator!=(const Iterator& it);
            bool operator==(const Iterator& it);

        private:
            Shape m_target_shape;
            Shape m_axis_walk_order;
            Coordinate m_coordinate;
            bool m_oob;
            bool m_empty;
        };

        Iterator begin() noexcept { return Iterator(m_target_shape); }
        Iterator end() noexcept { return Iterator(m_target_shape, true); }
    private:
        size_t index_source(const Coordinate& c) const;
        static Shape default_padding(size_t n_axes);
        static AxisVector default_axis_order(size_t n_axes);
        static Strides default_source_strides(size_t n_axes);
        static Coordinate default_source_start_corner(size_t n_axes);
        static Coordinate default_source_end_corner(const Shape& source_shape);

        Shape m_source_shape;
        Shape m_source_start_corner;
        Shape m_source_end_corner;
        Strides m_source_strides;
        AxisVector m_source_axis_order;
        Shape m_source_padding_below;
        Shape m_source_padding_above;

        Shape m_target_shape;
        size_t m_n_axes;
    };
}
