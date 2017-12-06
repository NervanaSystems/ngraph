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
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include "ngraph/common.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/except.hpp"

using namespace ngraph;

template <typename T>
inline T ceil_div(T x, T y)
{
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}

CoordinateTransform::CoordinateTransform(const Shape& source_space_shape,
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

    if (m_n_axes != source_start_corner.size())
    {
        throw std::domain_error(
            "Source start corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_end_corner.size())
    {
        throw std::domain_error(
            "Source end corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_strides.size())
    {
        throw std::domain_error(
            "Source strides do not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_axis_order.size())
    {
        // Note: this check is NOT redundant with the is_permutation check below, though you might think it is.
        // If the lengths don't match then is_permutation won't catch that; it'll either stop short or walk off
        // the end of source_axis_order.
        throw std::domain_error(
            "Source axis order does not have the same number of axes as the source space shape");
    }

    AxisVector all_axes(m_n_axes);
    size_t n = 0;
    std::generate(all_axes.begin(), all_axes.end(), [&n]() -> size_t { return n++; });

    if (!std::is_permutation(all_axes.begin(), all_axes.end(), source_axis_order.begin()))
    {
        throw std::domain_error(
            "Source axis order is not a permutation of {0,...,n-1} where n is the number of axes "
            "in the source space shape");
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (source_start_corner[i] >= source_space_shape[i] &&
            !(source_start_corner[i] == 0 && source_space_shape[i] == 0))
        {
            std::stringstream ss;

            ss << "The start corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (source_end_corner[i] > source_space_shape[i])
        {
            std::stringstream ss;

            ss << "The end corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (source_strides[i] == 0)
        {
            std::stringstream ss;

            ss << "The source stride is 0 at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        m_target_shape.push_back(ceil_div(source_end_corner[source_axis_order[axis]] -
                                              source_start_corner[source_axis_order[axis]],
                                          source_strides[source_axis_order[axis]]));
    }
}

AxisVector default_axis_order(size_t n_axes)
{
    AxisVector result(n_axes);
    size_t n = 0;
    std::generate(result.begin(), result.end(), [&n]() -> size_t { return n++; });

    return result;
}

CoordinateTransform::CoordinateTransform(const Shape& source_space_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides)
    : CoordinateTransform(source_space_shape,
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

CoordinateTransform::CoordinateTransform(const Shape& source_space_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner)
    : CoordinateTransform(source_space_shape,
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

CoordinateTransform::CoordinateTransform(const Shape& source_space_shape)
    : CoordinateTransform(source_space_shape,
                          default_source_start_corner(source_space_shape.size()),
                          default_source_end_corner(source_space_shape),
                          default_source_strides(source_space_shape.size()),
                          default_axis_order(source_space_shape.size()))
{
}

// Compute the index of a source-space coordinate in the buffer.
size_t CoordinateTransform::index_source(const Coordinate& c) const
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

// Compute the index of a target-space coordinate in thebuffer.
size_t CoordinateTransform::index(const Coordinate& c) const
{
    return index_source(to_source_coordinate(c));
}

// Convert a target-space coordinate to a source-space coordinate.
Coordinate CoordinateTransform::to_source_coordinate(const Coordinate& c) const
{
    if (c.size() != m_n_axes)
    {
        throw std::domain_error("Coordinate rank does not match the coordinate transform rank");
    }

    Coordinate result(c.size());

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        result[axis] = c[m_source_axis_order[axis]] * m_source_strides[m_source_axis_order[axis]] +
                       m_source_start_corner[m_source_axis_order[axis]];
    }

    return result;
}

// Check if a coordinate is in bounds of the target space.
bool CoordinateTransform::in_bounds(const Coordinate& c) const
{
    if (c.size() != m_n_axes)
    {
        return false;
    }

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        if (c[axis] < m_target_shape[axis] || c[axis] >= m_target_shape[axis])
        {
            return false;
        }
    }

    return true;
}

Coordinate CoordinateTransform::get_target_shape() const
{
    return m_target_shape;
}

// The "is_end" parameter is true if we want the "end()" iterator.
CoordinateTransform::Iterator::Iterator(const Shape& target_shape, bool is_end)
    : m_target_shape(target_shape)
{
    // Initial coordinate is (0,...,0) in the target space.
    m_coordinate = Coordinate(target_shape.size(), 0);

    // The case where we have a zero-length axis is a bit special, in that
    // the iterator always starts out of bounds.
    m_empty = false;

    for (auto s : target_shape)
    {
        if (s == 0)
        {
            m_empty = true;
            break;
        }
    }

    m_oob = is_end || m_empty;
}

void CoordinateTransform::Iterator::operator++()
{
    // If we are out of bounds, start over at (0,...0). (TODO: not sure if that's what we want. It might be best to stay put?)
    if (m_oob)
    {
        std::fill(m_coordinate.begin(), m_coordinate.end(), 0);
        m_oob = m_empty;
        return;
    }

    // Increment the target coordinate.
    for (size_t axis = m_target_shape.size(); axis-- > 0;)
    {
        m_coordinate[axis]++;

        if (m_coordinate[axis] < m_target_shape[axis])
        {
            // No carry-out, so we are done.
            return;
        }
        else
        {
            m_coordinate[axis] = 0;
        }
    }

    // If we are still here there was carry-out from the most significant axis. We are now out of bounds.
    m_oob = true;
}

CoordinateTransform::Iterator CoordinateTransform::Iterator::operator++(int)
{
    CoordinateTransform::Iterator temp = *this;
    ++(*this);
    return temp;
}

void CoordinateTransform::Iterator::operator+=(size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        ++(*this);
    }
}

Coordinate CoordinateTransform::Iterator::operator*()
{
    return m_coordinate;
}

bool CoordinateTransform::Iterator::operator!=(const Iterator& it)
{
    return !(*this == it);
}

bool CoordinateTransform::Iterator::operator==(const Iterator& it)
{
    if (m_target_shape != it.m_target_shape)
    {
        return false;
    }

    // Out-of-bounds iterators are always equal; in other words, an iterator is always equal to
    // end() even if the internally stored coordinates are different.
    if (m_oob && it.m_oob)
    {
        return true;
    }

    // If one iterator is out of bounds and the other is not, they are unequal even if their target
    // coordinates happen to match.
    if (m_oob != it.m_oob)
    {
        return false;
    }

    // Check axis-wise if the iterators are on the same target coordinate.
    for (size_t axis = 0; axis < m_target_shape.size(); axis++)
    {
        if (m_coordinate[axis] != it.m_coordinate[axis])
        {
            return false;
        }
    }

    return true;
}
