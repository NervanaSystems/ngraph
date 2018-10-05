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
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_static());
    ASSERT_EQ(size_t(ps.rank()), 0);
}

TEST(partial_shape, ps_construction_rank_dynamic)
{
    auto ps = PartialShape::dynamic();
    ASSERT_TRUE(ps.rank().is_dynamic());
    ASSERT_TRUE(ps.is_dynamic());
}

TEST(partial_shape, ps_construction_rank_static_shape_dynamic)
{
    auto ps = PartialShape{2, Dimension::dynamic(), 3};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_dynamic());
    ASSERT_EQ(size_t(ps.rank()), 3);
}

TEST(partial_shape, ps_construction_static)
{
    auto ps = PartialShape{2, 5, 3, 6};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_static());
    ASSERT_EQ(size_t(ps.rank()), 4);
}

TEST(partial_shape, dim_construction_static)
{
    Dimension dim{3};
    ASSERT_EQ(size_t(dim), 3);
    ASSERT_TRUE(dim.is_static());
}

TEST(partial_shape, dim_construction_dynamic)
{
    Dimension dim = Dimension::dynamic();
    ASSERT_TRUE(dim.is_dynamic());
}

TEST(partial_shape, dim_construction_size_t_max)
{
    EXPECT_ANY_THROW({ Dimension d{Dimension::s_dynamic_val}; });
}

TEST(partial_shape, dim_conversion_static)
{
    Dimension d{42};
    size_t s{d};
    ASSERT_EQ(s, 42);
}

TEST(partial_shape, dim_conversion_dynamic)
{
    EXPECT_ANY_THROW({
        size_t s{Dimension::dynamic()};

        s = 0; // Silence compiler warning about unused s
    });
}

TEST(partial_shape, rank_construction_static)
{
    Rank r{4};
    ASSERT_EQ(size_t(r), 4);
    ASSERT_TRUE(r.is_static());
}

TEST(partial_shape, rank_construction_dynamic)
{
    Rank r = Rank::dynamic();
    ASSERT_TRUE(r.is_dynamic());
}

TEST(partial_shape, dim_compatible_left_dynamic)
{
    Dimension d1{Dimension::dynamic()};
    Dimension d2{3};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_right_dynamic)
{
    Dimension d1{3};
    Dimension d2{Dimension::dynamic()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_dynamic)
{
    Dimension d1{Dimension::dynamic()};
    Dimension d2{Dimension::dynamic()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_static)
{
    Dimension d1{3};
    Dimension d2{8};
    Dimension d3{3};

    ASSERT_FALSE(d1.compatible(d2));
    ASSERT_TRUE(d1.compatible(d3));
}

TEST(partial_shape, shapes_compatible_both_rank_dynamic)
{
    PartialShape ps1{PartialShape::dynamic()};
    PartialShape ps2{PartialShape::dynamic()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_left_rank_dynamic)
{
    PartialShape ps1{3};
    PartialShape ps2{PartialShape::dynamic()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_right_rank_dynamic)
{
    PartialShape ps1{PartialShape::dynamic()};
    PartialShape ps2{4};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_all_known_equal)
{
    PartialShape ps1{2, Dimension::dynamic(), 3, Dimension::dynamic(), 5};
    PartialShape ps2{2, Dimension::dynamic(), Dimension::dynamic(), 4, 5};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_some_known_unequal)
{
    PartialShape ps1{2, Dimension::dynamic(), 3, Dimension::dynamic(), 5};
    PartialShape ps2{1, Dimension::dynamic(), Dimension::dynamic(), 4, 5};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_static_different_rank)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8, 10};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_static_same_rank_same_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_static_same_rank_different_dims)
{
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 3, 8};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, from_shape)
{
    Shape s{2, 4, 6, 8};
    PartialShape ps1{s};

    ASSERT_TRUE(ps1.rank().is_static());
    ASSERT_EQ(size_t(ps1.rank()), s.size());
    ASSERT_TRUE(ps1.is_static());
    ASSERT_EQ(size_t(ps1[0]), 2);
    ASSERT_EQ(size_t(ps1[1]), 4);
    ASSERT_EQ(size_t(ps1[2]), 6);
    ASSERT_EQ(size_t(ps1[3]), 8);
}

TEST(partial_shape, to_shape_static)
{
    PartialShape ps{2, 4, 6, 8};
    Shape s{ps.to_shape()};

    ASSERT_EQ(s, (Shape{2, 4, 6, 8}));
}

TEST(partial_shape, to_shape_dims_dynamic)
{
    PartialShape ps{2, 4, Dimension::dynamic(), 8};
    ASSERT_THROW({ ps.to_shape(); }, std::invalid_argument);
}

TEST(partial_shape, to_shape_rank_dynamic)
{
    PartialShape ps{PartialShape::dynamic()};
    ASSERT_THROW({ ps.to_shape(); }, std::invalid_argument);
}

TEST(partial_shape, tensor_descriptor_from_shape)
{
    descriptor::Tensor t{element::i32, Shape{1, 2, 3}, "Ankeny"};

    ASSERT_EQ(t.get_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, tensor_descriptor_from_static_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape{1, 2, 3}, "Burnside"};

    ASSERT_EQ(t.get_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, tensor_descriptor_from_rank_static_dynamic_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape{1, Dimension::dynamic(), 3}, "Couch"};

    ASSERT_EQ(size_t(t.get_partial_shape().rank()), 3);
    ASSERT_THROW({ t.get_shape(); }, std::invalid_argument);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(partial_shape, tensor_descriptor_from_rank_dynamic_partial_shape)
{
    descriptor::Tensor t{element::i32, PartialShape::dynamic(), "Davis"};

    ASSERT_TRUE(t.get_partial_shape().rank().is_dynamic());
    ASSERT_THROW({ t.get_shape(); }, std::invalid_argument);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape::dynamic()));
}

TEST(partial_shape, dim_same_scheme_both_dynamic)
{
    ASSERT_TRUE(Dimension::dynamic().same_scheme(Dimension::dynamic()));
}

TEST(partial_shape, dim_same_scheme_left_dynamic)
{
    ASSERT_FALSE(Dimension::dynamic().same_scheme(6));
}

TEST(partial_shape, dim_same_scheme_right_dynamic)
{
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension::dynamic()));
}

TEST(partial_shape, dim_same_scheme_both_static_same)
{
    ASSERT_TRUE(Dimension(6).same_scheme(Dimension(6)));
}

TEST(partial_shape, dim_same_scheme_both_static_different)
{
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension(7)));
}

TEST(partial_shape, partial_shape_same_scheme_both_dynamic)
{
    ASSERT_TRUE(PartialShape::dynamic().same_scheme(PartialShape::dynamic()));
}

TEST(partial_shape, partial_shape_same_scheme_left_dynamic_right_rank_static_dynamic)
{
    ASSERT_FALSE(PartialShape::dynamic().same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(partial_shape, partial_shape_same_scheme_left_dynamic_right_static)
{
    ASSERT_FALSE(PartialShape::dynamic().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_same_scheme_right_dynamic_left_rank_static_dynamic)
{
    ASSERT_FALSE((PartialShape{1, Dimension::dynamic(), 3}.same_scheme(PartialShape::dynamic())));
}

TEST(partial_shape, partial_shape_same_scheme_right_dynamic_left_static)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape::dynamic())));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_different_rank)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_different_rank)
{
    ASSERT_FALSE((PartialShape{1, Dimension::dynamic(), 3}.same_scheme(
        PartialShape{1, Dimension::dynamic(), 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_same_rank_different_dims)
{
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 3, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_same_rank_different_dims)
{
    ASSERT_FALSE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(
        PartialShape{1, 3, Dimension::dynamic()})));
}

TEST(partial_shape,
     partial_shape_same_scheme_both_rank_static_dynamic_same_rank_compatible_not_same)
{
    ASSERT_FALSE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(
        PartialShape{1, Dimension::dynamic(), 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_same_rank_compatible_same)
{
    ASSERT_TRUE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(
        PartialShape{1, 2, Dimension::dynamic()})));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_same_rank_same_dims)
{
    ASSERT_TRUE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_scalar)
{
    ASSERT_TRUE((PartialShape{}.same_scheme(PartialShape{})));
}

TEST(partial_shape, dim_merge_both_dynamic)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::dynamic(), Dimension::dynamic()));
    ASSERT_TRUE(d.is_dynamic());
}

TEST(partial_shape, dim_merge_left_dynamic)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::dynamic(), 3));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_right_dynamic)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, Dimension::dynamic()));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_both_static_equal)
{
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, 3));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(size_t(d), 3);
}

TEST(partial_shape, dim_merge_both_static_unequal)
{
    Dimension d = 163;
    ASSERT_FALSE(Dimension::merge(d, 3, 4));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(size_t(d), 163);
}

TEST(partial_shape, partial_shape_merge_both_rank_dynamic)
{
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.rank().is_dynamic());
}

TEST(partial_shape, partial_shape_merge_left_rank_dynamic_right_rank_static_dynamic)
{
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_left_rank_dynamic_right_static)
{
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_left_rank_static_dynamic_right_rank_dynamic)
{
    PartialShape s1{1, 2, Dimension::dynamic()};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_left_static_right_rank_dynamic)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_consistent)
{
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic(), Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_same_rank_inconsistent)
{
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{2, 2, Dimension::dynamic(), Dimension::dynamic()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_different_rank)
{
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_static_consistent)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_static_inconsistent)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_static_different_rank)
{
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, dim_pluseq_left_dynamic)
{
    Dimension d1{Dimension::dynamic()};
    Dimension d2{2};

    d1 += d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_pluseq_right_dynamic)
{
    Dimension d1{2};
    Dimension d2{Dimension::dynamic()};

    d1 += d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_pluseq_both_static)
{
    Dimension d1{3};
    Dimension d2{2};

    d1 += d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(size_t(d1), 5);
}

TEST(partial_shape, dim_timeseq_left_dynamic_right_nonzero)
{
    Dimension d1{Dimension::dynamic()};
    Dimension d2{2};

    d1 *= d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_timeseq_left_dynamic_right_zero)
{
    Dimension d1{Dimension::dynamic()};
    Dimension d2{0};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(size_t(d1), 0);
}

TEST(partial_shape, dim_timeseq_right_dynamic_left_nonzero)
{
    Dimension d1{2};
    Dimension d2{Dimension::dynamic()};

    d1 *= d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_timeseq_right_dynamic_left_zero)
{
    Dimension d1{0};
    Dimension d2{Dimension::dynamic()};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(size_t(d1), 0);
}

TEST(partial_shape, dim_timeseq_both_static)
{
    Dimension d1{3};
    Dimension d2{2};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(size_t(d1), 6);
}
