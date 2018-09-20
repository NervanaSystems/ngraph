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

#include <memory>
using namespace std;
using namespace ngraph;

#define EXPECT_HAS_SUBSTRING(haystack, needle)                                                     \
    EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

//
// Tests for broadcast.
//
TEST(type_prop, broadcast_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, batchnorm_rank_less_than_2)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{1});
    try
    {
        auto bc = make_shared<op::BatchNorm>(0.001, dummy, dummy, dummy);
        FAIL() << "BatchNorm c-tor should throw for tensors whose rank is less than 2";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input argument must have rank of at least 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_zero_channel_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{1, 0, 2, 3});
    try
    {
        auto bc = make_shared<op::BatchNorm>(0.001, dummy, dummy, dummy);
        FAIL() << "BatchNorm c-tor should throw for tensors w/ zero-dimension channels";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input argument's channel dimension must have size of at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_et_check)
{
    auto dummy_f32 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy_f64 = make_shared<op::Parameter>(element::f64, Shape{3});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNorm>(0.001, dummy_f32, dummy_f64, param);
        FAIL() << "BatchNorm c-tor should throw for different element types";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Element type of beta"));
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("is not equal to the element type of input"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_shape_check)
{
    auto dummy_3 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy_4 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNorm>(0.001, dummy_4, dummy_3, param);
        FAIL() << "BatchNorm c-tor should throw if gamma and beta shapes don't match";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of gamma must match the channel dimension of the input data"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_backprop_4d_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{});
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto bc =
            make_shared<op::BatchNormBackprop>(0.001, dummy, dummy, param, dummy, dummy, dummy);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input data shape is not a 4D tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_backprop_et_check)
{
    auto dummy_f32 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy_f64 = make_shared<op::Parameter>(element::f64, Shape{3});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNormBackprop>(
            0.001, dummy_f32, dummy_f64, param, dummy_f32, dummy_f32, dummy_f32);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Element type of beta"));
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("is not equal to the element type of input"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_backprop_shape_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc =
            make_shared<op::BatchNormBackprop>(0.001, dummy, dummy2, param, dummy2, dummy2, dummy2);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of beta must match the channel dimension of the input data"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_backprop_delta_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 3});

    try
    {
        auto bc =
            make_shared<op::BatchNormBackprop>(0.001, dummy, dummy, param, dummy, dummy, delta);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape of delta must match the shape of the input data"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_wrong_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32,
                                             Shape{
                                                 2, 2,
                                             });
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Not all arguments have the same rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dimensions of argument 2 do not match for axis 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Concatenation axis (3) is out of bounds (inputs have rank 3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 12});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 2);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Not all arguments have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, convert_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto c = make_shared<op::Convert>(param, element::i32);
    ASSERT_EQ(c->get_element_type(), element::i32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 4}));
}

TEST(type_prop, dot_deduce_scalar_2d)
{
    // Deduce type for scalar/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_2d_scalar)
{
    // Deduce type for matrix/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_scalar_scalar)
{
    // Deduce type for scalar/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_scalar_1d)
{
    // Deduce type for scalar/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{6}));
}

TEST(type_prop, dot_deduce_1d)
{
    // Deduce type for vector/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_2d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 3}));
}

TEST(type_prop, dot_deduce_different_rank)
{
    // Deduce type for different-rank tensor arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 8, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 8, 4, 1, 3}));
}

TEST(type_prop, dot_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments do not have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dot reduction axes size mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Paired axes (axis 1 from arg0, axis 0 from arg1) do not have same length"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Tests for binary elementwise ops.
//
void test_binary(std::string node_type,
                 shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Argument 0 shape Shape{2, 4} differs in shape from argument 1"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_bad_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                            const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument 0 element type element::Type{32, 1, "
                                             "1,float} differs in element type from argument 1"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->get_arguments()[0]));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, add_bad_arguments)
{
    test_binary("Add",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Add>(x, y);
                });
}

TEST(type_prop, divide_bad_arguments)
{
    test_binary("Divide",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Divide>(x, y);
                });
}

TEST(type_prop, multiply_bad_arguments)
{
    test_binary("Multiply",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Multiply>(x, y);
                });
}

TEST(type_prop, subtract_bad_arguments)
{
    test_binary("Subtract",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Subtract>(x, y);
                });
}

//
// Tests for binary elementwise logical ops.
//
void test_binary_logical(std::string node_type,
                         shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_2_4_param_3 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::boolean, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Argument 0 shape Shape{2, 4} differs in shape from argument 1"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_differ_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                               const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument 0 element type element::Type{8, 0, 1,char} "
                                             "differs in element type from argument 1"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    auto test_binary_non_bool_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                                 const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), "must have boolean element type");
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }

    };

    test_binary_differ_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);
    test_binary_non_bool_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_0);
    test_binary_non_bool_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_3);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->get_arguments()[0]));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, and_bad_arguments)
{
    test_binary_logical(
        "And", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::And>(x, y);
        });
}

TEST(type_prop, or_bad_arguments)
{
    test_binary_logical(
        "Or", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::Or>(x, y);
        });
}

TEST(type_prop, comparison_good)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto eq = make_shared<op::Equal>(tv0_2_4_param_0, tv0_2_4_param_1);
    EXPECT_EQ(eq->get_element_type(), element::boolean);
    EXPECT_EQ(eq->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, binary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Add>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Negative>(tv0_2_4_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not all have the same shape"));
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not all have the same shape"));
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not all have the same shape"));
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
    catch (const NodeValidationError& error)
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments 1 and 2 do not have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::f32);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::f32);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::f32);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::f32);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, reduce_nonscalar)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect non-scalar initial value for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_elem_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::boolean, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect element type mismatch for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for reductee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_return_element_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Equal>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element return type for reduction function";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Return element type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_return_shape_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(f_param_0 + f_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect return shape for reduction function";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg0_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument 0 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg1_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::boolean, Shape{});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument 1 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg_count_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1 + f_param_2,
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Reduction function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction axis is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, function_call_deduce)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B * C), op::ParameterVector{A, B, C});

    // Now make "f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto r = make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z});
    auto r_p_r = r + r;

    ASSERT_EQ(r_p_r->get_element_type(), element::f32);
    ASSERT_EQ(r_p_r->get_shape(), shape);
}

TEST(type_prop, reshape_deduce_s2v)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_v2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_t2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2v_01)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_m2v_10)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_t2v_012)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_t2v_120)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_not_enough_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Not enough axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_too_many_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0, 3}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_duplicate_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{3, 3, 3});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Product of output shape dimensions does not match "
                                         "product of argument shape dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sl = make_shared<op::Slice>(param, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{3}));
}

TEST(type_prop, slice_deduce_matrix)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, slice_deduce_matrix_strided)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 2});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{1, 3}));
}

TEST(type_prop, slice_deduce_matrix_strided_uneven)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 4});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{1, 2}));
}

TEST(type_prop, slice_deduce_vector_edge)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6}));
}

TEST(type_prop, slice_deduce_matrix_edge)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, slice_deduce_matrix_zero_cols)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6, 0}));
}

TEST(type_prop, slice_deduce_matrix_zero_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{0, 0}));
}

TEST(type_prop, slice_deduce_vector_invalid_strides)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Rank of strides (2) does not match rank of argument (1)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 0 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_missing)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of lower bounds (1) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_missing)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of upper bounds (1) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_extra)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of lower bounds (3) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_extra)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of upper bounds (3) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scalar_constant_deduce_float32)
{
    auto c = op::Constant::create(element::f32, Shape{}, {208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool)
{
    auto c = op::Constant::create(element::boolean, Shape{}, {1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32)
{
    auto c = op::Constant::create(element::f32, Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool)
{
    auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count)
{
    try
    {
        auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Did not get the expected number of literals for a "
                                         "constant of shape Shape{2, 2} (got 3, expected 1 or 4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{1, 3});
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 2});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided_uneven)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 4});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_edge)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix_edge)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_cols)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 0});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_zero)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 0});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_invalid_strides)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto sl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{0}, Coordinate{7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Rank of strides (2) does not match rank of argument (1)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_rank_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6, 5});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_element_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{1, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape of replacement tensor (Shape{3, 6}) does not match "
                                         "the slice shape (Shape{4, 6})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch_strided)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{1, 1}, Coordinate{5, 7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Shape of replacement tensor (Shape{4, 6}) does not match the slice shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{7});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 0 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 9});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 0});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_missing)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of lower bounds (1) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_missing)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of upper bounds (1) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_extra)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of lower bounds (3) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_extra)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank of upper bounds (3) does not match rank of argument (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_scalar)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{});
    auto oh = make_shared<op::OneHot>(param, Shape{9}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9}));
}

TEST(type_prop, one_hot_deduce_vector_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{9, 8}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9, 8}));
}

TEST(type_prop, one_hot_deduce_vector_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{8, 9}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{8, 9}));
}

TEST(type_prop, one_hot_deduce_matrix_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{2, 12, 24}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{2, 12, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 2, 24}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 2, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_2)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 2}, 2);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 2}));
}

TEST(type_prop, one_hot_deduce_floating_point)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 2);
    ASSERT_EQ(oh->get_element_type(), element::f32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 8}));
}

TEST(type_prop, one_hot_deduce_axis_oob)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected.";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("One-hot axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_shape_incompatible)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 22, 8}, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible one-hot output shape not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shape Shape{12, 24} does not match the expected shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_1d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         Strides{1},
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            Strides{1},
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 48}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_uneven)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 5};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_uneven)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 5};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 3}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_even)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 6};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_even)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 6};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 82}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 87}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 285}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{3});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});   // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, 3};
    auto padding_above = CoordinateDiff{3, 4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 138}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, 3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, 4}));
}

TEST(type_prop, conv_2d_deduce_padded_neg)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, -3};
    auto padding_above = CoordinateDiff{3, -4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 124}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, -3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, -4}));
}

TEST(type_prop, conv_2d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46, 44}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 37, 38}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_data_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto padding_below = CoordinateDiff{0, 0};
    auto padding_above = CoordinateDiff{0, 0};
    auto data_dilate_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 86, 137}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_data_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto padding_below = CoordinateDiff{0, 0, 0};
    auto padding_above = CoordinateDiff{0, 0, 0};
    auto data_dilate_strides = Strides{2, 3, 2};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 5, 6, 5}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3, 2}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_invalid_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 3, 3, 3});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{3, 3, 2, 2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Element types for data batch and filters do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch input must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_1d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch input must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_2d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch input must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_batch_size)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_input_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 0, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 input channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_many)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many filter dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Filter input must have rank equal to the data batch"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_few)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few filter dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Filter input must have rank equal to the data batch"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_output_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 output channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Output channel count for filters is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_channel_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with channel count mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input channel count for filters (3) does not match the "
                                         "number of channels in the data batch (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Rank of window movement strides does not match the number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong window dilation stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Rank of window dilation strides does not match the number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong data dilation stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Rank of data dilation strides does not match the number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_below_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0, 0},
                                                 CoordinateDiff{0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-below rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Rank of the padding below does not match the number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_above_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-above rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Rank of the padding above does not match the number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_negative_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{0, 0},
                                                 Strides{0, 0},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-7, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with negative-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input dimension after padding and dilation is non-positive"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_zero_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{0, 0},
                                                 Strides{0, 0},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-6, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input dimension after padding and dilation is non-positive"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input dimension after padding and dilation is non-positive"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 0});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Filters shape at spatial dimension 1 is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length window dilation stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window dilation stride at spatial dimension 1 is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length data dilation stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data dilation stride at spatial dimension 1 is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilated_window_too_large)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{1, 1}, Strides{4, 4});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized dilated window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Post-dilation window shape is smaller than the "
                                         "post-padding/dilation input item shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{0, 1});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window movement stride at spatial dimension 0 is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{2, 3, 2}));
}

TEST(type_prop, max_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data input shape does not have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data input shape does not have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data input shape does not have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window shape rank does not match number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window shape rank does not match number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window movement stride rank does not match number of spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_input_data_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data input spatial dimension 0 has zero length even after padding"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Window shape dimension 1 has zero length"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window shape after padding is larger than the spatial dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window movement strides dimension 0 has zero length"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_0d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{}));
}

TEST(type_prop, reverse_1d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_1d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_2d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_3d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_2)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_02)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_12)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_012)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_oob)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    try
    {
        auto rev = make_shared<op::Reverse>(param, AxisSet{0, 3, 2});

        // Should have thrown, so fail if it didn't
        FAIL() << "Axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reverse axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_1_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{4, 4});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for seq_lenghts whose rank isn't equal to 1";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Sequence indices must be a 1-dimensional tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_batch_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 3;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds batch axis index";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch axis index (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_sequence_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 1;
        size_t seq_axis = 3;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds sequence axis index";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Sequence axis index (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_seq_len_size_equal_to_batch_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw when sequence length size isn't equal to "
                  "batch dimension";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence length (3) is not equal to batch axis dimension (4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_1d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{1};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{13}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_even)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4}));
}

TEST(type_prop, reduce_window_deduce_2d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2};
    Strides move_strides{4, 3};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4, 3}));
}

TEST(type_prop, reduce_window_deduce_3d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4, 3, 6}));
}

TEST(type_prop, reduce_window_deduce_non_scalar_init)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{3});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar initial value not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_different_element_types)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::i32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Different element types not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for reductee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_bad_window_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Bad window shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has different rank from input tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_bad_move_strides)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Bad window movement strides not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window movement strides have different rank from input tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 0, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has a zero-length axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_zero_length_stride)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 0, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window movement stride not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window movement stride for some axis is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_window_too_big)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 11, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Window too big not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction window is bigger than input"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1,
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Too many reduction function parameters not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Reduction function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_0_wrong_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 0 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of reduction function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_0_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 0 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_1_wrong_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 1 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of reduction function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_1_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 1 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(NodeVector{f_param_0 + f_param_1, f_param_0 * f_param_1},
                                   op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multiple-output reduction function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output reduction function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_reduction_function_return_element_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Convert>(f_param_0 + f_param_1, element::i32),
                                   op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Reduction function return element type mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Return element type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_reduction_function_return_shape_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(f_param_0 + f_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Reduction function return shape mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_1d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4};
    Strides move_strides{1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16}));
}

TEST(type_prop, select_and_scatter_deduce_2d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13, 14});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 5};
    Strides move_strides{1, 1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18}));
}

TEST(type_prop, select_and_scatter_deduce_3d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13, 14, 9});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 5, 2};
    Strides move_strides{1, 1, 1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_3d_strided)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 3, 2};
    Strides move_strides{4, 6, 5};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_3d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_init_not_scalar)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{4});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar init value not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_init_elem_type_wrong)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::i32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect init element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for selectee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_elem_type_wrong)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::i32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect source tensor element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for selectee and source tensors do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_shape_wrong_rank)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong window shape rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window shape has different rank from selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_strides_wrong_rank)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong window strides rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window movement strides have different rank from selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_shape_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 0, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window shape axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has a zero-length axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_strides_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 0, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window strides axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window movement stride for some axis is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_too_big)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 19, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Window too big not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction window is bigger than selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_tensor_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong source tensor shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Source tensor does not have expected shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function parameter count not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Selection function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_0_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_1, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for selection function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of selection function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_0_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_1, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for selection function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_1_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_0),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for selection function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of selection function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_1_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_0),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for selection function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        std::vector<std::shared_ptr<Node>>{make_shared<op::Greater>(f_param_0, f_param_1),
                                           make_shared<op::Greater>(f_param_0, f_param_1)},
        op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multi-output selection function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output selection function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_result_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function result element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return element type from selection function is not boolean"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_result_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(
            make_shared<op::Greater>(f_param_0, f_param_1), Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function result type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(g_param_0 + g_param_1,
                                   op::ParameterVector{g_param_0, g_param_1, g_param_2});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function parameter count not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Scatter function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_0_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_1 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for scatter function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of scatter function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_0_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_1 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for scatter function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_1_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_0, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for scatter function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of scatter function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_1_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto g =
        make_shared<Function>(g_param_0 + g_param_0, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for scatter function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(
        std::vector<std::shared_ptr<Node>>{g_param_0 + g_param_1, g_param_0 + g_param_1},
        op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multi-output scatter function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output scatter function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_result_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Greater>(g_param_0, g_param_1),
                                   op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function result element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return element type from scatter function does "
                              "not match the init value type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_result_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(
        make_shared<op::Broadcast>(g_param_0 + g_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function result shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Return shape from scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_padded_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{5, 6, 4};
    Shape padding_above{6, 4, 5};
    auto avg_pool = make_shared<op::AvgPool>(
        param, window_shape, move_strides, padding_below, padding_above, true);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 9, 6, 5}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{5, 6, 4}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{6, 4, 5}));
}

TEST(type_prop, avg_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data input shape does not have rank of at least 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data input shape does not have rank of at least 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data input shape does not have rank of at least 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch size is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Channel count is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window shape rank does not match number of spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window shape rank does not match number of spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Window movement stride rank does not match number of spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_below_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2, 3};
    Shape padding_above{1, 2};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong below-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Below-padding rank does not match number of spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_above_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2};
    Shape padding_above{1, 2, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong above-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Above-padding rank does not match number of spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_input_item_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data input spatial dimension 0 has zero length even after padding");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Window shape dimension 1 has zero length");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window shape after padding is larger than the spatial dimensions");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Window movement strides dimension 0 has zero length");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_1d_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{2};
    Shape padding_above{3};
    Shape padding_interior{0};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{55}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{2}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{3}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{0}));
}

TEST(type_prop, pad_deduce_1d_interior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{2};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{148}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{0}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{0}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2}));
}

TEST(type_prop, pad_deduce_1d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5};
    Shape padding_above{6};
    Shape padding_interior{2};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2}));
}

TEST(type_prop, pad_deduce_2d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3};
    Shape padding_above{6, 9};
    Shape padding_interior{2, 3};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159, 169}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5, 3}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6, 9}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2, 3}));
}

TEST(type_prop, pad_deduce_3d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159, 169, 24}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5, 3, 0}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6, 9, 4}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2, 3, 0}));
}

TEST(type_prop, pad_deduce_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Element tpye mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_nonscalar_pad_value)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar pad value not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument for padding value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_below_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0, 6};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong below-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank for padding below does not match the rank of the data argument"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_above_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong above-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank for padding above does not match the rank of the data argument"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_interior_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0, 9, 3};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong interior padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank for interior padding does not match the rank of the data argument"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, sum_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    auto r0 = make_shared<op::Sum>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::f32);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Sum>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::f32);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Sum>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::f32);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Sum>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::f32);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, sum_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Sum>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for sum";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (2) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_scalar)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 0, element::i32);
        FAIL() << "ArgMin c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument rank must be at least 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 2, element::i32);
        FAIL() << "ArgMin c-tor should throw for axis out of bounds";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "is greater than rank of");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_index_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 1, element::f32);
        FAIL() << "ArgMin c-tor should throw for invalid index type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element type must be");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::i32, 1, true);
        FAIL() << "TopK c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input Tensor's rank must be greater than 0");
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "TopK axis must be less than rank");
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element type must be i64 or i32");
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
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "K should not exceed TopK axis length");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
