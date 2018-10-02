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
    ASSERT_EQ(size_t(ps.rank()), 0);
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
    ASSERT_EQ(size_t(ps.rank()), 3);
}

TEST(partial_shape, ps_construction_complete)
{
    auto ps = PartialShape{2, 5, 3, 6};
    ASSERT_TRUE(ps.rank_is_determined());
    ASSERT_TRUE(ps.rank().is_determined());
    ASSERT_TRUE(ps.is_complete());
    ASSERT_EQ(size_t(ps.rank()), 4);
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
    EXPECT_ANY_THROW({ Dimension d{Dimension::s_undetermined_val}; });
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

TEST(partial_shape, dim_compatible_left_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{3};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_right_undetermined)
{
    Dimension d1{3};
    Dimension d2{Dimension::undetermined()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_undetermined)
{
    Dimension d1{Dimension::undetermined()};
    Dimension d2{Dimension::undetermined()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_determined)
{
    Dimension d1{3};
    Dimension d2{8};
    Dimension d3{3};

    ASSERT_FALSE(d1.compatible(d2));
    ASSERT_TRUE(d1.compatible(d3));
}

TEST(partial_shape, shapes_compatible_both_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_left_rank_undetermined)
{
    PartialShape ps1{3};
    PartialShape ps2{PartialShape::undetermined()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_right_rank_undetermined)
{
    PartialShape ps1{PartialShape::undetermined()};
    PartialShape ps2{4};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_all_known_equal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{2, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_some_known_unequal)
{
    PartialShape ps1{2, Dimension::undetermined(), 3, Dimension::undetermined(), 5};
    PartialShape ps2{1, Dimension::undetermined(), Dimension::undetermined(), 4, 5};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_complete_different_rank)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8, 10};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_complete_same_rank_same_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_complete_same_rank_different_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 3, 8};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, from_shape)
{
    Shape s{2, 4, 6, 8};
    PartialShape ps1{s};

    // TODO(amprocte): No way to examine contents of ps1 yet.
    ASSERT_TRUE(ps1.is_complete());
    ASSERT_TRUE(ps1.rank_is_determined());
    ASSERT_EQ(size_t(ps1.rank()), s.size());
}

TEST(partial_shape, to_shape_complete)
{
    PartialShape ps{2, 4, 6, 8};
    Shape s{ps.to_shape()};

    ASSERT_EQ(s, (Shape{2, 4, 6, 8}));
}

TEST(partial_shape, to_shape_dims_undetermined)
{
    PartialShape ps{2, 4, Dimension::undetermined(), 8};
    ASSERT_THROW({ ps.to_shape(); }, std::invalid_argument);
}

TEST(partial_shape, to_shape_rank_undetermined)
{
    PartialShape ps{PartialShape::undetermined()};
    ASSERT_THROW({ ps.to_shape(); }, std::invalid_argument);
}

TEST(partial_shape, tensor_descriptor_from_shape)
{
    descriptor::Tensor t{element::i32, Shape{1, 2, 3}, "Ankeny"};

    ASSERT_EQ(t.get_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
}

TEST(partial_shape, tensor_descriptor_from_complete_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape{1, 2, 3}, "Burnside"};

    ASSERT_EQ(t.get_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
}

TEST(partial_shape, tensor_descriptor_from_incomplete_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape{1, Dimension::undetermined(), 3}, "Couch"};

    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
    ASSERT_THROW({ t.get_shape(); }, std::invalid_argument);
}

TEST(partial_shape, tensor_descriptor_from_rankless_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape::undetermined(), "Davis"};

    ASSERT_FALSE(t.get_partial_shape().rank().is_determined());
    ASSERT_THROW({ t.get_shape(); }, std::invalid_argument);
}

TEST(partial_shape, dim_same_scheme_both_undetermined)
{
    ASSERT_TRUE(Dimension::undetermined().same_scheme(Dimension::undetermined()));
}

TEST(partial_shape, dim_same_scheme_left_undetermined)
{
    ASSERT_FALSE(Dimension::undetermined().same_scheme(6));
}

TEST(partial_shape, dim_same_scheme_right_undetermined)
{
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension::undetermined()));
}

TEST(partial_shape, dim_same_scheme_both_determined_same)
{
    ASSERT_TRUE(Dimension(6).same_scheme(Dimension(6)));
}

TEST(partial_shape, dim_same_scheme_both_determined_different)
{
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension(7)));
}

TEST(partial_shape, partial_shape_same_scheme_both_undetermined)
{
    ASSERT_TRUE(PartialShape::undetermined().same_scheme(PartialShape::undetermined()));
}

TEST(partial_shape, partial_shape_same_scheme_left_undetermined_right_incomplete)
{
    ASSERT_FALSE(
        PartialShape::undetermined().same_scheme(PartialShape{1, Dimension::undetermined(), 3}));
}

TEST(partial_shape, partial_shape_same_scheme_left_undetermined_right_complete)
{
    ASSERT_FALSE(PartialShape::undetermined().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_same_scheme_right_undetermined_left_incomplete)
{
    ASSERT_FALSE(
        (PartialShape{1, Dimension::undetermined(), 3}.same_scheme(PartialShape::undetermined())));
}

TEST(partial_shape, partial_shape_same_scheme_right_undetermined_left_complete)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape::undetermined())));
}

TEST(partial_shape, partial_shape_same_scheme_both_complete_different_rank)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_incomplete_different_rank)
{
    ASSERT_FALSE((PartialShape{1, Dimension::undetermined(), 3}.same_scheme(
        PartialShape{1, Dimension::undetermined(), 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_complete_same_rank_different_dims)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 3, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_incomplete_same_rank_different_dims)
{
    ASSERT_FALSE((PartialShape{1, 2, Dimension::undetermined()}.same_scheme(
        PartialShape{1, 3, Dimension::undetermined()})));
}

TEST(partial_shape, partial_shape_same_scheme_both_incomplete_same_rank_compatible_not_same)
{
    ASSERT_FALSE((PartialShape{1, 2, Dimension::undetermined()}.same_scheme(
        PartialShape{1, Dimension::undetermined(), 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_incomplete_same_rank_compatible_same)
{
    ASSERT_TRUE((PartialShape{1, 2, Dimension::undetermined()}.same_scheme(
        PartialShape{1, 2, Dimension::undetermined()})));
}

TEST(partial_shape, partial_shape_same_scheme_both_complete_same_rank_same_dims)
{
    ASSERT_TRUE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_scalar)
{
    ASSERT_TRUE((PartialShape{}.same_scheme(PartialShape{})));
}

TEST(partial_shape, dim_merge_both_undetermined)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::undetermined(), Dimension::undetermined()));
    ASSERT_FALSE(d.is_determined());
}

TEST(partial_shape, dim_merge_left_undetermined)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::undetermined(), 3));
    ASSERT_TRUE(d.is_determined());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_right_undetermined)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, Dimension::undetermined()));
    ASSERT_TRUE(d.is_determined());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_both_determined_equal)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, 3));
    ASSERT_TRUE(d.is_determined());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_both_determined_unequal)
{
    Dimension d = 163;
    ASSERT_FALSE(Dimension::merge(d, 3, 4));
    ASSERT_TRUE(d.is_determined());
    ASSERT_EQ(size_t(d), 163);
}

TEST(partial_shape, partial_shape_merge_both_undetermined)
{
    PartialShape s1{PartialShape::undetermined()};
    const PartialShape s2{PartialShape::undetermined()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_FALSE(s1.rank().is_determined());
}

TEST(partial_shape, partial_shape_merge_left_undetermined_right_incomplete)
{
    PartialShape s1{PartialShape::undetermined()};
    const PartialShape s2{1, 2, Dimension::undetermined()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::undetermined()}));
}

TEST(partial_shape, partial_shape_merge_left_undetermined_right_complete)
{
    PartialShape s1{PartialShape::undetermined()};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_left_incomplete_right_undetermined)
{
    PartialShape s1{1, 2, Dimension::undetermined()};
    const PartialShape s2{PartialShape::undetermined()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::undetermined()}));
}

TEST(partial_shape, partial_shape_merge_left_complete_right_undetermined)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{PartialShape::undetermined()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_incomplete_consistent)
{
    PartialShape s1{1, Dimension::undetermined(), 3, Dimension::undetermined()};
    const PartialShape s2{1, 2, Dimension::undetermined(), Dimension::undetermined()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3, Dimension::undetermined()}));
}

TEST(partial_shape, partial_shape_merge_both_incomplete_same_rank_inconsistent)
{
    PartialShape s1{1, Dimension::undetermined(), 3, Dimension::undetermined()};
    const PartialShape s2{2, 2, Dimension::undetermined(), Dimension::undetermined()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_incomplete_different_rank)
{
    PartialShape s1{1, Dimension::undetermined(), 3, Dimension::undetermined()};
    const PartialShape s2{1, 2, Dimension::undetermined()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_complete_consistent)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_complete_inconsistent)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_complete_different_rank)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}
