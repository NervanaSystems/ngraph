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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

TEST(partial_shape, ps_construction_empty)
{
    auto ps = PartialShape{};
    ASSERT_TRUE(ps.rank_is_determined());
    ASSERT_TRUE(ps.rank().is_determined());
    ASSERT_TRUE(ps.is_complete());
    ASSERT_EQ(ps.rank(), 0);
}

TEST(partial_shape, ps_construction_undetermined)
{
    auto ps = PartialShape::undetermined();
    ASSERT_FALSE(ps.rank_is_determined());
    ASSERT_FALSE(ps.rank().is_determined());
    ASSERT_FALSE(ps.is_complete());
}

TEST(partial_shape, ps_construction_incomplete)
{
    auto ps = PartialShape{2, Dimension::undetermined(), 3};
    ASSERT_TRUE(ps.rank_is_determined());
    ASSERT_TRUE(ps.rank().is_determined());
    ASSERT_FALSE(ps.is_complete());
    ASSERT_EQ(ps.rank(), 3);
}

TEST(partial_shape, ps_construction_complete)
{
    auto ps = PartialShape{2, 5, 3, 6};
    ASSERT_TRUE(ps.rank_is_determined());
    ASSERT_TRUE(ps.rank().is_determined());
    ASSERT_TRUE(ps.is_complete());
    ASSERT_EQ(ps.rank(), 4);
}

TEST(partial_shape, dim_construction_determined)
{
    Dimension dim{3};
    ASSERT_EQ(size_t(dim), 3);
    ASSERT_TRUE(dim.is_determined());
}

TEST(partial_shape, dim_construction_undetermined)
{
    Dimension dim = Dimension::undetermined();
    ASSERT_FALSE(dim.is_determined());
}

TEST(partial_shape, dim_construction_size_t_max)
{
    EXPECT_ANY_THROW({ Dimension d{std::numeric_limits<size_t>::max()}; });
}

TEST(partial_shape, dim_conversion_determined)
{
    Dimension d{42};
    size_t s{d};
    ASSERT_EQ(s, 42);
}

TEST(partial_shape, dim_conversion_undetermined)
{
    EXPECT_ANY_THROW({
        size_t s{Dimension::undetermined()};

        s = 0; // Silence compiler warning about unused s
    });
}

TEST(partial_shape, rank_construction_determined)
{
    Rank r{4};
    ASSERT_EQ(size_t(r), 4);
    ASSERT_TRUE(r.is_determined());
}

TEST(partial_shape, rank_construction_undetermined)
{
    Rank r = Rank::undetermined();
    ASSERT_FALSE(r.is_determined());
}

TEST(partial_shape, dim_equal_left_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{3};

    ASSERT_FALSE(d1 == d2);
    ASSERT_TRUE(d1.possibly_eq(d2));
}

TEST(partial_shape, dim_not_equal_left_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{3};

    ASSERT_FALSE(d1 != d2);
    ASSERT_TRUE(d1.possibly_neq(d2));
}

TEST(partial_shape, dim_equal_right_undetermined)
{
    Dimension d1{3};
    Dimension d2{Dimension::undetermined()};

    ASSERT_FALSE(d1 == d2);
    ASSERT_TRUE(d1.possibly_eq(d2));
}

TEST(partial_shape, dim_not_equal_right_undetermined)
{
    Dimension d1{3};
    Dimension d2{Dimension::undetermined()};

    ASSERT_FALSE(d1 != d2);
    ASSERT_TRUE(d1.possibly_neq(d2));
}

TEST(partial_shape, dim_equal_both_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{Dimension::undetermined()};

    ASSERT_FALSE(d1 == d2);
    ASSERT_TRUE(d1.possibly_eq(d2));
}

TEST(partial_shape, dim_not_equal_both_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{Dimension::undetermined()};

    ASSERT_FALSE(d1 != d2);
    ASSERT_TRUE(d1.possibly_neq(d2));
}

TEST(partial_shape, dim_equal_both_determined)
{
    Dimension d1{3};
    Dimension d2{8};
    Dimension d3{3};

    ASSERT_FALSE(d1 == d2);
    ASSERT_FALSE(d1.possibly_eq(d2));
    ASSERT_TRUE(d1 == d3);
    ASSERT_TRUE(d1.possibly_eq(d3));
}

TEST(partial_shape, dim_not_equal_both_determined)
{
    Dimension d1{3};
    Dimension d2{8};
    Dimension d3{3};

    ASSERT_TRUE(d1 != d2);
    ASSERT_TRUE(d1.possibly_neq(d2));
    ASSERT_FALSE(d1 != d3);
    ASSERT_FALSE(d1.possibly_neq(d3));
}

TEST(partial_shape, shapes_equal_both_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_TRUE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_left_rank_undetermined)
{
    PartialShape ps1{3};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_TRUE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_right_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{4};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_TRUE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_both_partial_all_known_equal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{2, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_TRUE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_both_partial_some_known_unequal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{1, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_FALSE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_both_complete_different_rank)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8, 10};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_FALSE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_both_complete_same_rank_same_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8};

    ASSERT_TRUE(ps1 == ps2);
    ASSERT_TRUE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_equal_both_complete_same_rank_different_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 3, 8};

    ASSERT_FALSE(ps1 == ps2);
    ASSERT_FALSE(ps1.possibly_eq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_FALSE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_left_rank_undetermined)
{
    PartialShape ps1{3};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_FALSE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_right_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{4};

    ASSERT_FALSE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_partial_all_known_equal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{2, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_FALSE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_partial_some_known_unequal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{1, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_TRUE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_complete_different_rank)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8, 10};

    ASSERT_TRUE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_complete_same_rank_same_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8};

    ASSERT_FALSE(ps1 != ps2);
    ASSERT_FALSE(ps1.possibly_neq(ps2));
}

TEST(partial_shape, shapes_not_equal_both_complete_same_rank_different_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 3, 8};

    ASSERT_TRUE(ps1 != ps2);
    ASSERT_TRUE(ps1.possibly_neq(ps2));
}
