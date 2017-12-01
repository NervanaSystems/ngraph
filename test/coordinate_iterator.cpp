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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

TEST(coordinate_iterator, construct)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 1, 1, 1};
    Coordinate window_outer_corner{2, 3, 5, 6};
    Coordinate window_inner_corner{0, 0, 0, 0};

    auto ci = CoordinateIterator(space_shape, strides, window_outer_corner, window_inner_corner);
}

TEST(coordinate_iterator, construct_defaults)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{2, 2, 2, 1};

    auto ci = CoordinateIterator(space_shape, strides);
}

TEST(coordinate_iterator, construct_defaults_stride)
{
    Shape space_shape{2, 3, 5, 6};

    auto ci = CoordinateIterator(space_shape);
}

TEST(coordinate_iterator, construct_bad_outer_oob)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 1, 1, 1};
    Coordinate window_outer_corner{2, 4, 5, 6};
    Coordinate window_inner_corner{0, 0, 0, 0};

    EXPECT_ANY_THROW({
        auto ci =
            CoordinateIterator(space_shape, strides, window_outer_corner, window_inner_corner);
    });
}

TEST(coordinate_iterator, construct_bad_inner_oob)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 1, 1, 1};
    Coordinate window_outer_corner{2, 3, 5, 6};
    Coordinate window_inner_corner{0, 3, 0, 0};

    EXPECT_ANY_THROW({
        auto ci =
            CoordinateIterator(space_shape, strides, window_outer_corner, window_inner_corner);
    });
}

TEST(coordinate_iterator, construct_bad_inner_outside_outer)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 1, 1, 1};
    Coordinate window_outer_corner{2, 1, 5, 6};
    Coordinate window_inner_corner{0, 2, 0, 0};

    EXPECT_ANY_THROW({
        auto ci =
            CoordinateIterator(space_shape, strides, window_outer_corner, window_inner_corner);
    });
}

TEST(coordinate_iterator, construct_bad_zero_stride)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 0, 1, 1};
    Coordinate window_outer_corner{2, 3, 5, 6};
    Coordinate window_inner_corner{0, 0, 0, 0};

    EXPECT_ANY_THROW({
        auto ci =
            CoordinateIterator(space_shape, strides, window_outer_corner, window_inner_corner);
    });
}

TEST(coordinate_iterator, cover_count_defaults)
{
    Shape space_shape{2, 3, 5, 6};

    auto ci = CoordinateIterator(space_shape);

    size_t count = 0;
    size_t expected_index = 0;

    do
    {
        count++;
        EXPECT_EQ(ci.get_current_index(), expected_index);
        expected_index++;
    } while (ci.increment());

    EXPECT_EQ(count, 2 * 3 * 5 * 6);
}

TEST(coordinate_iterator, cover_count_stride_2)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 1, 1, 2};

    auto ci = CoordinateIterator(space_shape, strides);

    size_t count = 0;
    size_t expected_index = 0;

    do
    {
        count++;
        EXPECT_EQ(ci.get_current_index(), expected_index);
        expected_index += 2;
    } while (ci.increment());

    EXPECT_EQ(count, 2 * 3 * 5 * 6 / 2);
}

#define CEIL_DIV(x, y) (1 + (((x)-1) / (y)))

TEST(coordinate_iterator, cover_count_stride_uneven)
{
    Shape space_shape{2, 3, 5, 6};
    Strides strides{1, 2, 2, 3};

    auto ci = CoordinateIterator(space_shape, strides);

    size_t count = 0;

    do
    {
        count++;
    } while (ci.increment());

    EXPECT_EQ(count, CEIL_DIV(2, 1) * CEIL_DIV(3, 2) * CEIL_DIV(5, 2) * CEIL_DIV(6, 3));
}
