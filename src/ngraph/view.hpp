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
    class View
    {
    public:
        View(const Shape& space_shape,
             const Coordinate& start_corner,
             const Coordinate& end_corner,
             const Strides& strides,
             const AxisVector& axis_storage_order,
             const AxisVector& axis_walk_order);

        View(const Shape& space_shape,
             const Coordinate& start_corner,
             const Coordinate& end_corner,
             const Strides& strides);

        View(const Shape& space_shape,
             const Coordinate& start_corner,
             const Coordinate& end_corner);

        View(const Shape& space_shape);

        size_t index(const Coordinate& c) const;
        bool in_bounds(const Coordinate& c) const;
        Coordinate to_raw(const Coordinate& c) const;
        Coordinate get_virtual_shape() const;

        class Iterator
        {
        public:
            Iterator(const Shape& virtual_shape, const Shape& axis_walk_order, bool is_end = false);

            void operator++();
            Coordinate operator*();
            bool operator!=(const Iterator& it);
            bool operator==(const Iterator& it);

        private:
            Shape m_virtual_shape;
            Shape m_axis_walk_order;
            Coordinate m_coordinate;
            bool m_oob;
            bool m_empty;
        };

        Iterator begin() noexcept { return Iterator(m_virtual_shape, m_axis_walk_order); }
        Iterator end() noexcept { return Iterator(m_virtual_shape, m_axis_walk_order, true); }
    private:
        size_t index_raw(const Coordinate& c) const;

        Shape m_space_shape;
        Shape m_start_corner;
        Shape m_end_corner;
        Shape m_strides;
        Shape m_virtual_shape;
        AxisVector m_axis_storage_order;
        AxisVector m_axis_walk_order;
        size_t m_n_axes;
    };

    Coordinate project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes);
    Coordinate inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val);
}
