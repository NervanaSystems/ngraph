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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

#include "ngraph/common.hpp"
#include "ngraph/except.hpp"
#include "ngraph/view.hpp"

using namespace ngraph;

template <typename T>
inline T ceil_div(T x, T y)
{
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}

View::View(const Shape& space_shape,
           const Coordinate& start_corner,
           const Coordinate& end_corner,
           const Strides& strides,
           const AxisVector& axis_storage_order,
           const AxisVector& axis_walk_order)
    : m_space_shape(space_shape)
    , m_start_corner(start_corner)
    , m_end_corner(end_corner)
    , m_strides(strides)
    , m_axis_storage_order(axis_storage_order)
    , m_axis_walk_order(axis_walk_order)
{
    m_n_axes = space_shape.size();

    // In the real thing we won't use assert.
    assert(m_n_axes == start_corner.size());
    assert(m_n_axes == end_corner.size());
    assert(m_n_axes == strides.size());
    assert(m_n_axes == axis_storage_order.size());
    assert(m_n_axes == axis_walk_order.size());

    AxisVector all_axes(m_n_axes);
    size_t n = 0;
    std::generate(all_axes.begin(), all_axes.end(), [&n]() -> size_t { return n++; });
    assert(std::is_permutation(all_axes.begin(), all_axes.end(), axis_storage_order.begin()));
    assert(std::is_permutation(all_axes.begin(), all_axes.end(), axis_walk_order.begin()));

    assert(std::all_of(all_axes.begin(), all_axes.end(), [space_shape, start_corner](size_t i) {
        return (start_corner[i] < space_shape[i] || (start_corner[i] == 0 && space_shape[i] == 0));
    }));
    assert(std::all_of(all_axes.begin(), all_axes.end(), [space_shape, end_corner](size_t i) {
        return (end_corner[i] <= space_shape[i]);
    }));
    assert(std::all_of(all_axes.begin(), all_axes.end(), [start_corner, end_corner](size_t i) {
        return (start_corner[i] <= end_corner[i]);
    }));
    assert(std::all_of(strides.begin(), strides.end(), [](size_t x) { return x > 0; }));

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        m_virtual_shape.push_back(ceil_div(end_corner[axis] - start_corner[axis], strides[axis]));
    }
}

AxisVector default_axis_order(size_t n_axes)
{
    AxisVector result(n_axes);
    size_t n = 0;
    std::generate(result.begin(), result.end(), [&n]() -> size_t { return n++; });

    return result;
}

View::View(const Shape& space_shape,
           const Coordinate& start_corner,
           const Coordinate& end_corner,
           const Strides& strides)
    : View(space_shape,
           start_corner,
           end_corner,
           strides,
           default_axis_order(space_shape.size()),
           default_axis_order(space_shape.size()))
{
}

Strides default_strides(size_t n_axes)
{
    return AxisVector(n_axes, 1);
}

View::View(const Shape& space_shape, const Coordinate& start_corner, const Coordinate& end_corner)
    : View(space_shape,
           start_corner,
           end_corner,
           default_strides(space_shape.size()),
           default_axis_order(space_shape.size()),
           default_axis_order(space_shape.size()))
{
}

Coordinate default_start_corner(size_t n_axes)
{
    return Coordinate(n_axes, 0);
}

Coordinate default_end_corner(const Shape& space_shape)
{
    return space_shape;
}

View::View(const Shape& space_shape)
    : View(space_shape,
           default_start_corner(space_shape.size()),
           default_end_corner(space_shape),
           default_strides(space_shape.size()),
           default_axis_order(space_shape.size()),
           default_axis_order(space_shape.size()))
{
}

size_t View::index_raw(const Coordinate& c) const
{
    size_t index = 0;
    size_t stride = 1;

    for (size_t axis = m_n_axes; axis-- > 0;)
    {
        index += c[axis] * stride;
        stride *= m_space_shape[axis];
    }

    return index;
}

size_t View::index(const Coordinate& c) const
{
    return index_raw(to_raw(c));
}

Coordinate View::to_raw(const Coordinate& c) const
{
    assert(c.size() == m_n_axes);

    Coordinate result(c.size());

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        result[axis] = c[m_axis_storage_order[axis]] * m_strides[m_axis_storage_order[axis]] +
                       m_start_corner[m_axis_storage_order[axis]];
    }

    return result;
}

bool View::in_bounds(const Coordinate& c) const
{
    if (c.size() != m_n_axes)
    {
        return false;
    }

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        if (c[axis] < m_virtual_shape[axis] || c[axis] >= m_virtual_shape[axis])
        {
            return false;
        }
    }

    return true;
}

Coordinate View::get_virtual_shape() const
{
    return m_virtual_shape;
}

View::Iterator::Iterator(const Shape& virtual_shape, const Shape& axis_walk_order, bool is_end)
    : m_virtual_shape(virtual_shape)
    , m_axis_walk_order(axis_walk_order)
    , m_oob(is_end)
{
    m_coordinate = Coordinate(virtual_shape.size(), 0);

    m_empty = false;

    for (auto s : virtual_shape)
    {
        if (s == 0)
        {
            m_empty = true;
            break;
        }
    }

    m_oob = m_oob || m_empty;
}

void View::Iterator::operator++()
{
    if (m_oob)
    {
        std::fill(m_coordinate.begin(), m_coordinate.end(), 0);
        m_oob = m_empty;
        return;
    }

    for (size_t axis = m_virtual_shape.size(); axis-- > 0;)
    {
        m_coordinate[m_axis_walk_order[axis]]++;

        if (m_coordinate[m_axis_walk_order[axis]] < m_virtual_shape[m_axis_walk_order[axis]])
        {
            return;
        }
        else
        {
            m_coordinate[m_axis_walk_order[axis]] = 0;
        }
    }

    m_oob = true;
}

Coordinate View::Iterator::operator*()
{
    return m_coordinate;
}

bool View::Iterator::operator!=(const Iterator& it)
{
    return !(*this == it);
}

bool View::Iterator::operator==(const Iterator& it)
{
    if (m_virtual_shape != it.m_virtual_shape)
    {
        return false;
    }

    if (m_axis_walk_order != it.m_axis_walk_order)
    {
        return false;
    }

    if (m_oob && it.m_oob)
    {
        return true;
    }

    if (m_oob != it.m_oob)
    {
        return false;
    }

    for (size_t axis = 0; axis < m_virtual_shape.size(); axis++)
    {
        if (m_coordinate[axis] != it.m_coordinate[axis])
        {
            return false;
        }
    }

    return true;
}

// TODO: check validity, i.e. that all deleted_axes are valid
Coordinate ngraph::project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes)
{
    Coordinate result;

    for (size_t i = 0; i < coord.size(); i++)
    {
        if (deleted_axes.find(i) == deleted_axes.end())
        {
            result.push_back(coord[i]);
        }
    }

    return result;
}

// TODO: for the moment, just one axis at a time, please. Later could pass in an std::map from axis positions to axis lengths.
// TODO: check validity, i.e. that the new axis is < coord_size+1.
Coordinate
    ngraph::inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val)
{
    Coordinate result;

    size_t original_pos = 0;

    for (size_t result_pos = 0; result_pos < coord.size() + 1; result_pos++)
    {
        if (result_pos == new_axis_pos)
        {
            result.push_back(new_axis_val);
        }
        else
        {
            result.push_back(coord[original_pos++]);
        }
    }

    return result;
}
