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

#include <cstdio>
#include <iostream>
#include <vector>

#include "ngraph/common.hpp"
#include "ngraph/coordinate_iterator.hpp"
#include "ngraph/except.hpp"

using namespace ngraph;

CoordinateIterator::CoordinateIterator(const Shape& space_shape,
                                       const Strides& strides,
                                       const Coordinate& window_outer_corner,
                                       const Coordinate& window_inner_corner)
    : m_space_shape(space_shape)
    , m_strides(strides)
    , m_window_outer_corner(window_outer_corner)
    , m_window_inner_corner(window_inner_corner)
    , m_current_coordinate(window_inner_corner)
{
    if (space_shape.size() != window_inner_corner.size())
    {
        throw ngraph_error("Coordinate iterator inner corner rank does not match space shape rank");
    }

    if (space_shape.size() != window_outer_corner.size())
    {
        throw ngraph_error("Coordinate iterator outer corner rank does not match space shape rank");
    }

    if (space_shape.size() != strides.size())
    {
        throw ngraph_error("Coordinate iterator stride rank does not match space shape rank");
    }

    for (size_t i = 0; i < space_shape.size(); i++)
    {
        if (window_inner_corner[i] > window_outer_corner[i])
        {
            throw ngraph_error("Coordinate iterator inner corner is outside outer corner");
        }

        if (window_inner_corner[i] >= m_space_shape[i])
        {
            throw ngraph_error("Coordinate iterator inner corner is out of bounds");
        }

        if (window_outer_corner[i] > m_space_shape[i])
        {
            throw ngraph_error("Coordinate iterator outer corner is out of bounds");
        }

        if (m_strides[i] == 0)
        {
            throw ngraph_error("Coordinate iterator stride is zero");
        }
    }
}

CoordinateIterator::CoordinateIterator(const Shape& space_shape)
    : CoordinateIterator(space_shape,
                         Strides(space_shape.size(), 1),
                         space_shape,
                         Coordinate(space_shape.size(), 0))
{
}

CoordinateIterator::CoordinateIterator(const Shape& space_shape, const Strides& strides)
    : CoordinateIterator(space_shape, strides, space_shape, Coordinate(space_shape.size(), 0))
{
}

size_t CoordinateIterator::get_current_index() const
{
    size_t index = 0;
    size_t stride = 1;

    for (size_t i = m_space_shape.size(); i-- > 0;)
    {
        index += m_current_coordinate[i] * stride;
        stride *= m_space_shape[i];
    }

    return index;
}

bool CoordinateIterator::increment()
{
    bool overflow = true;

    for (size_t i = m_space_shape.size(); i-- > 0;)
    {
        m_current_coordinate[i] += m_strides[i];
        if (m_current_coordinate[i] >= m_window_outer_corner[i])
        {
            m_current_coordinate[i] = m_window_inner_corner[i];
        }
        else
        {
            overflow = false;
            break;
        }
    }

    return !overflow;
}
