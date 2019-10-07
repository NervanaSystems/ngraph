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

TEST(type_prop, dynreshape_arg_static_pattern_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_arg_static_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_zero)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 0, 2, 8});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 0, 32});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{1, 2, 0, 32}));

    auto r2 = make_shared<op::DynReshape>(arg, pattern, true /*zero_flag*/);
    EXPECT_EQ(r2->get_output_shape(0), (Shape{1, 2, 2, 32}));

    auto r3 = make_shared<op::DynReshape>(dynamic_arg, pattern, true /*zero_flag*/);
    EXPECT_TRUE(
        r3->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic(), 32}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 4, -1});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{1, 2, 4, 16}));

    auto r2 = make_shared<op::DynReshape>(dynamic_arg, pattern);
    EXPECT_TRUE(
        r2->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 4, Dimension::dynamic()}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_zero_negative)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 2, 0});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{2}, {0, -1});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    auto r2 = make_shared<op::DynReshape>(arg, pattern, true);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{0, 0}));
    EXPECT_EQ(r2->get_output_shape(0), (Shape{2, 0}));

    auto r3 = make_shared<op::DynReshape>(dynamic_arg, pattern);
    auto r4 = make_shared<op::DynReshape>(dynamic_arg, pattern, true);
    EXPECT_TRUE(r3->get_output_partial_shape(0).same_scheme(PartialShape{0, Dimension::dynamic()}));
    EXPECT_TRUE(r4->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative_failure1)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, -1, -1});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Expected failure on dynreshape construction";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("More than one dimension has size of -1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative_failure2)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 4, -2});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Expected failure on dynreshape construction";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dim size cannot be less than -1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

void DynReshape_Test_Shape_Except(const shared_ptr<Node>& param_0, const shared_ptr<Node>& param_1)
{
    try
    {
        auto r = make_shared<op::DynReshape>(param_0, param_1);
        FAIL() << "Did not detect parameter shape not rank 1";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreshape_arg_static_pattern_static_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_static_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_static_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_pattern_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_pattern_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::boolean, Shape{4});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Did not detect pattern elment type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Pattern must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_v1_arg_rank_static_pattern_zero)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 0, 2, 8});
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 0, 32});

    auto reshape_v1_static = make_shared<op::v1::Reshape>(arg, pattern, true);
    EXPECT_EQ(reshape_v1_static->get_output_shape(0), Shape({1, 2, 2, 32}));

    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto reshape_v1_dynamic = make_shared<op::v1::Reshape>(dynamic_arg, pattern, true);
    EXPECT_TRUE(reshape_v1_dynamic->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 2, Dimension::dynamic(), 32}));
}
