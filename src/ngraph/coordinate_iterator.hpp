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

#include <cstdio>
#include <iostream>
#include <vector>

#include "ngraph/common.hpp"

namespace ngraph
{
    class CoordinateIterator
    {
    public:
        CoordinateIterator(const Shape& space_shape,
                           const Strides& strides,
                           const Coordinate& window_outer_corner,
                           const Coordinate& window_inner_corner);

        CoordinateIterator(const Shape& space_shape);

        CoordinateIterator(const Shape& space_shape, const Strides& strides);

        Coordinate get_current_coordinate() const { return m_current_coordinate; }
        size_t get_current_index() const;
        bool increment();

    private:
        const Shape m_space_shape;
        const Strides m_strides;
        const Coordinate m_window_outer_corner;
        const Coordinate m_window_inner_corner;

        Coordinate m_current_coordinate;
    };
}
