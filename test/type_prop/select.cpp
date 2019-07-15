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

TEST(type_prop, select_deduce)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_shape_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{3, 5});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_b)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_c)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 0 does not have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_bc)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_et_dynamic_arg1_arg2_et_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    try
    {
        auto sel = make_shared<op::Select>(param0, param1, param2);
        FAIL() << "Did not detect mismatched element types for args 1 and 2 (element type-dynamic "
                  "arg0)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::dynamic);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_arg0_rank_dynamic_static_arg1_arg2_rank_dynamic_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::boolean, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_arg1_rank_dynamic_static_arg0_arg2_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_arg2_rank_dynamic_static_arg0_arg1_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_all_rank_static_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3});

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).is_static());
    ASSERT_EQ(sel->get_output_shape(0), (Shape{2, 8, 3}));
}

TEST(type_prop, select_partial_all_rank_static_intransitive_incompatibility)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 3});

    try
    {
        auto sel = make_shared<op::Select>(param0, param1, param2);
        FAIL() << "Did not detect intransitive partial-shape incompatibility";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
