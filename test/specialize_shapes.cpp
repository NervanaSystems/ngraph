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
#include "ngraph/specialize_shapes.hpp"

using namespace ngraph;

// Simple case: create a function with static parameter shapes and "specialize" them to the same
// shapes.
TEST(specialize_shapes, et_shape_static)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    auto g = specialize_shapes(
        f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of dynamic element types.
TEST(specialize_shapes, et_dynamic_shape_static)
{
    auto p0 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    auto g = specialize_shapes(
        f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-dynamic shapes.
TEST(specialize_shapes, et_static_shape_rank_dynamic)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    auto g = specialize_shapes(
        f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-static dynamic shapes.
TEST(specialize_shapes, et_static_shape_rank_static_dynamic)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic(3));

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    auto g = specialize_shapes(
        f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-dynamic shapes to a case where validation will fail.
//
// (The input shapes we provide at specialization time are inconsistent.)
TEST(specialize_shapes, et_static_shape_rank_dynamic_validation_fails)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(
                f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3, 4}});
        },
        NodeValidationFailure);
}

// Test specialization of dynamic element types to a case where validation will fail.
//
// (The input element types we provide at specialization time are inconsistent.)
TEST(specialize_shapes, et_dynamic_shape_static_validation_fails)
{
    auto p0 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(
                f, {element::u32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});
        },
        NodeValidationFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have the wrong
// rank.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use NGRAPH_CHECK, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_shapes, et_static_shape_rank_static_dynamic_rank_mismatch)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic(3));

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(
                f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 2, 3, 4}});
        },
        CheckFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have wrong
// dimensions.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use NGRAPH_CHECK, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_shapes, et_static_shape_rank_static_dynamic_dim_mismatch)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 =
        std::make_shared<op::Parameter>(element::i32, PartialShape{1, Dimension::dynamic(), 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(
                f, {element::f32, element::i32}, {PartialShape{1, 2, 3}, PartialShape{1, 9, 4}});
        },
        CheckFailure);
}

// Test for failure when we supply the wrong number of replacement element types.
TEST(specialize_shapes, et_count_wrong)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(f,
                              {element::f32, element::i32, element::u32},
                              {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}});
        },
        CheckFailure);
}

// Test for failure when we supply the wrong number of replacement shapes.
TEST(specialize_shapes, shape_count_wrong)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    ASSERT_THROW(
        {
            specialize_shapes(
                f,
                {element::f32, element::i32},
                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}, PartialShape{4, 5, 6}});
        },
        CheckFailure);
}
