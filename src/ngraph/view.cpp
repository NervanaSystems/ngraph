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

View::View(const Shape& source_space_shape,
           const Coordinate& source_start_corner,
           const Coordinate& source_end_corner,
           const Strides& source_strides,
           const AxisVector& source_axis_order)
    : m_source_space_shape(source_space_shape)
    , m_source_start_corner(source_start_corner)
    , m_source_end_corner(source_end_corner)
    , m_source_strides(source_strides)
    , m_source_axis_order(source_axis_order)
{
    m_n_axes = source_space_shape.size();

    // In the real thing we won't use assert.
    assert(m_n_axes == source_start_corner.size());
    assert(m_n_axes == source_end_corner.size());
    assert(m_n_axes == source_strides.size());
    assert(m_n_axes == source_axis_order.size());

    AxisVector all_axes(m_n_axes);
    size_t n = 0;
    std::generate(all_axes.begin(), all_axes.end(), [&n]() -> size_t { return n++; });
    assert(std::is_permutation(all_axes.begin(), all_axes.end(), source_axis_order.begin()));

    assert(std::all_of(all_axes.begin(), all_axes.end(), [source_space_shape, source_start_corner](size_t i) {
        return (source_start_corner[i] < source_space_shape[i] || (source_start_corner[i] == 0 && source_space_shape[i] == 0));
    }));
    assert(std::all_of(all_axes.begin(), all_axes.end(), [source_space_shape, source_end_corner](size_t i) {
        return (source_end_corner[i] <= source_space_shape[i]);
    }));
    assert(std::all_of(all_axes.begin(), all_axes.end(), [source_start_corner, source_end_corner](size_t i) {
        return (source_start_corner[i] <= source_end_corner[i]);
    }));
    assert(std::all_of(source_strides.begin(), source_strides.end(), [](size_t x) { return x > 0; }));

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        m_virtual_shape.push_back(ceil_div(source_end_corner[source_axis_order[axis]] - source_start_corner[source_axis_order[axis]], source_strides[source_axis_order[axis]]));
    }
}

AxisVector default_axis_order(size_t n_axes)
{
    AxisVector result(n_axes);
    size_t n = 0;
    std::generate(result.begin(), result.end(), [&n]() -> size_t { return n++; });

    return result;
}

View::View(const Shape& source_space_shape,
           const Coordinate& source_start_corner,
           const Coordinate& source_end_corner,
           const Strides& source_strides)
    : View(source_space_shape,
           source_start_corner,
           source_end_corner,
           source_strides,
           default_axis_order(source_space_shape.size()))
{
}

Strides default_source_strides(size_t n_axes)
{
    return AxisVector(n_axes, 1);
}

View::View(const Shape& source_space_shape, const Coordinate& source_start_corner, const Coordinate& source_end_corner)
    : View(source_space_shape,
           source_start_corner,
           source_end_corner,
           default_source_strides(source_space_shape.size()),
           default_axis_order(source_space_shape.size()))
{
}

Coordinate default_source_start_corner(size_t n_axes)
{
    return Coordinate(n_axes, 0);
}

Coordinate default_source_end_corner(const Shape& source_space_shape)
{
    return source_space_shape;
}

View::View(const Shape& source_space_shape)
    : View(source_space_shape,
           default_source_start_corner(source_space_shape.size()),
           default_source_end_corner(source_space_shape),
           default_source_strides(source_space_shape.size()),
           default_axis_order(source_space_shape.size()))
{
}

size_t View::index_raw(const Coordinate& c) const
{
    size_t index = 0;
    size_t stride = 1;

    for (size_t axis = m_n_axes; axis-- > 0;)
    {
        index += c[axis] * stride;
        stride *= m_source_space_shape[axis];
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
        result[axis] = c[m_source_axis_order[axis]] * m_source_strides[m_source_axis_order[axis]] +
                       m_source_start_corner[m_source_axis_order[axis]];
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

View::Iterator::Iterator(const Shape& virtual_shape, bool is_end)
    : m_virtual_shape(virtual_shape)
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
        m_coordinate[axis]++;

        if (m_coordinate[axis] < m_virtual_shape[axis])
        {
            return;
        }
        else
        {
            m_coordinate[axis] = 0;
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
