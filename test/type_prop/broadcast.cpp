//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, broadcast_deduce)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v1)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});

    auto bc = make_shared<op::v1::Broadcast>(param, target_shape, op::AutoBroadcastType::NUMPY);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 6}));
}

TEST(type_prop, broadcast_v1_axes_mapping)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});

    auto bc = make_shared<op::v1::Broadcast>(param, target_shape, axes_mapping);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 1}));
}

TEST(type_prop, broadcast_v1_fail_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{3}, {1, 2, 3});

    try
    {
        auto bc = make_shared<op::v1::Broadcast>(param, target_shape, axes_mapping);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast axes_mapping shape Shape{3} doesn't match rank of input tensor 2");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v1_fail_transpose)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 1, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 1});

    try
    {
        auto bc = make_shared<op::v1::Broadcast>(param, target_shape, axes_mapping);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast doesn't permit transposes. axes_mapping AxisVector{2, 1} "
                             "not in sorted order");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v1_fail_axes_map)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 3});

    try
    {
        auto bc = make_shared<op::v1::Broadcast>(param, target_shape, axes_mapping);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes_mapping[1]: 3 exceeds target rank 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v1_fail_axes_map_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});

    try
    {
        auto bc = make_shared<op::v1::Broadcast>(param, target_shape, axes_mapping);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast target[axes_mapping[1]] Expected 1. Got 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
