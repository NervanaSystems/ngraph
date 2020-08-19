//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

TEST(type_prop, gather_no_axis)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v0::Gather>(P, I);
    ASSERT_EQ(G->get_output_element_type(0), element::f32);
    ASSERT_EQ(G->get_output_shape(0), out_shape);
}

TEST(type_prop, gather)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v0::Gather>(P, I, 1);
    ASSERT_EQ(G->get_output_element_type(0), element::f32);
    ASSERT_EQ(G->get_output_shape(0), out_shape);
}

TEST(type_prop, gather_fail_params_rank)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::v0::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::v0::Gather>(P, I, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("params rank is expected to be at least axis + 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_fail_indices_element_type)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    try
    {
        auto G = make_shared<op::v0::Gather>(P, I, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_v1_incorrect_axis_shape)
{
    auto params = make_shared<op::v0::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<op::v0::Parameter>(element::i64, Shape{2});
    try
    {
        auto G = make_shared<op::v1::Gather>(params, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect axis input shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Axes input must be scalar or have 1 element (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_v1_axis_out_of_input_rank)
{
    auto params = make_shared<op::v0::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    try
    {
        auto G = make_shared<op::v1::Gather>(params, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect element of axis input";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The axis must => 0 and <= input_rank (axis:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_v1_negative_axis)
{
    auto params = make_shared<op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto indices = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    int64_t axis = -2;
    auto axis_node = make_shared<op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto gather_v1 = make_shared<op::v1::Gather>(params, indices, axis_node);
    ASSERT_EQ(gather_v1->get_axis(), 1);
}
