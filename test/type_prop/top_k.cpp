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

TEST(type_prop, topk_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::i32, 1, true);
        FAIL() << "TopK c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument rank must be greater than 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_top_k)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 2, element::i32, 1, true);
        FAIL() << "TopK c-tor should throw for invalid top k axis";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "TopK axis (2) is out of bounds");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_index_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::f32, 1, true);
        FAIL() << "TopK c-tor should throw for invalid index element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_k)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::i32, 3, true);
        FAIL() << "TopK c-tor should throw for invalid K";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "K (3) exceeds the dimension (2) of the TopK axis (axis 0)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(topk->get_output_partial_shape(1).rank().is_dynamic());
    ASSERT_TRUE(topk->get_sort() == op::TopK::SortType::NONE);
}

TEST(type_prop, topk_rank_dynamic_result_et_dynamic)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::dynamic};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Dynamic result element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument element type must not be dynamic");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_dynamic_result_et_invalid)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Invalid result element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_known_topk_dim_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 999;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 999, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 999, Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_topk_dim_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 0;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_axis_oob)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "TopK axis out-of-bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_axis_oob)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 0;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "TopK axis out-of-bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_known_too_big)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 4;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Oversize K not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 0;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_k_known_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 2;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic()}));
}
