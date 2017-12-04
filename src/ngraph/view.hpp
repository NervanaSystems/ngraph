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
        View(const Shape& source_shape,
             const Coordinate& source_start_corner,
             const Coordinate& source_end_corner,
             const Strides& source_strides,
             const AxisVector& source_axis_order);

        View(const Shape& source_shape,
             const Coordinate& source_start_corner,
             const Coordinate& source_end_corner,
             const Strides& source_strides);

        View(const Shape& source_shape,
             const Coordinate& source_start_corner,
             const Coordinate& source_end_corner);

        View(const Shape& source_space_shape);

        size_t index(const Coordinate& c) const;
        bool in_bounds(const Coordinate& c) const;
        Coordinate get_target_shape() const;

        class Iterator
        {
        public:
            Iterator(const Shape& target_shape, bool is_end = false);

            void operator++();
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
        Coordinate to_source_coordinate(const Coordinate& c) const;
        size_t index_source(const Coordinate& c) const;

        Shape m_source_space_shape;
        Shape m_source_start_corner;
        Shape m_source_end_corner;
        Strides m_source_strides;
        AxisVector m_source_axis_order;

        Shape m_target_shape;
        size_t m_n_axes;
    };

    Coordinate project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes);
    Shape project_shape(const Shape& shape, const AxisSet& deleted_axes);

    Coordinate inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val);
    Shape inject_shape(const Shape& shape, size_t new_axis_pos, size_t new_axis_length);
}
