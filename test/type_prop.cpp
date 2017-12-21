// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

#include <memory>
using namespace std;
using namespace ngraph;

//
// Tests for broadcast.
//
TEST(type_prop, broadcast_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 4});
    auto bc = make_shared<op::Broadcast>(param, Shape{2, 3, 4}, AxisSet{1});
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{2, 3, 4}));
}

TEST(type_prop, broadcast_deduce_incorrect)
{
    // Check deduced type against incorrectly specified type
    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Broadcast>(param, Shape{2, 4, 3}, AxisSet{1});
        bc->assert_value_type(element::Float32::element_type(), Shape{2, 3, 4});

        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Broadcast arg, shape, and axes are incompatible"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_bad_arguments)
{
    try
    {
        // Check for bad arguments
        auto param = make_shared<op::Parameter>(make_shared<TupleType>());
        auto bc = make_shared<op::Broadcast>(param, Shape{2, 4, 3}, AxisSet{1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Tuple argument to broadcast not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Arguments for node type \"Broadcast\" must be tensor views"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 1);
    auto c_vt = c->get_value_type();
    ASSERT_EQ(*c_vt, TensorViewType(element::Float32::element_type(), Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_incorrect)
{
    // Check deduced type against incorrectly specified type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 1);
        c->assert_value_type(element::Float32::element_type(), Shape{2, 14, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Setting value type to a different ValueType"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_rank)
{
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(),
                                             Shape{
                                                 2, 2,
                                             });
    try
    {
        auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments to concat do not have same rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape)
{
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Arguments to concat do not have same dimension on a non-concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob)
{
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Concatenation axis is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 8});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 12});
    auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 2);
    auto c_vt = c->get_value_type();
    ASSERT_EQ(*c_vt, TensorViewType(element::Float32::element_type(), Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::Int32::element_type(), Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(Nodes{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, convert_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    auto c = make_shared<op::Convert>(param, element::Int32::element_type());
    auto c_vt = c->get_value_type();
    ASSERT_EQ(*c_vt, TensorViewType(element::Int32::element_type(), Shape{2, 3, 4}));
}

TEST(type_prop, convert_deduce_incorrect)
{
    // Check deduced type against incorrectly specified type
    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3, 4});
    try
    {
        auto c = make_shared<op::Convert>(param, element::Int32::element_type());
        c->assert_value_type(element::Int32::element_type(), Shape{2, 14, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Setting value type to a different ValueType"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_deduce_scalar_2d)
{
    // Deduce type for scalar/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 5});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{4, 5}));
}

TEST(type_prop, dot_deduce_2d_scalar)
{
    // Deduce type for matrix/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 5});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{4, 5}));
}

TEST(type_prop, dot_deduce_scalar_scalar)
{
    // Deduce type for scalar/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, dot_deduce_scalar_1d)
{
    // Deduce type for scalar/vector arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{6}));
}

TEST(type_prop, dot_deduce_1d)
{
    // Deduce type for vector/vector arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, dot_deduce_2d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{4, 3}));
}

TEST(type_prop, dot_deduce_different_rank)
{
    // Deduce type for different-rank tensor arguments
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 8, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 1, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{2, 8, 4, 1, 3}));
}

TEST(type_prop, dot_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::Int32::element_type(), Shape{2, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments to dot must have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dot reduction axes size mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Dot axes do not have same length"));
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
    auto tp0_param = make_shared<op::Parameter>(make_shared<TupleType>());
    auto tp1_param = make_shared<op::Parameter>(make_shared<TupleType>());
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), Shape{2, 4}));
    auto tv0_4_2_param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 2}));

    auto test_binary_bad_arguments_tuple = [&](const shared_ptr<Node>& x,
                                               const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            //node->get_value_type();
            // Should have thrown, so fail if it didn't
            FAIL() << "Tuple argument not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_EQ(
                error.what(),
                std::string("Arguments for node type \"" + node_type + "\" must be tensor views"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_tuple(tp0_param, tp1_param);
    test_binary_bad_arguments_tuple(tp0_param, tv0_2_4_param_0);
    test_binary_bad_arguments_tuple(tv0_2_4_param_0, tp0_param);
    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            node->get_value_type();
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_EQ(error.what(), std::string("Arguments must have the same tensor view shape"));
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
            node->get_value_type();
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_EQ(error.what(),
                      std::string("Arguments must have the same tensor view element type"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_EQ(*node->get_value_type(), *node->get_input_ops()[0]->get_value_type());
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

TEST(type_prop, comparison_good)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto eq = make_shared<op::Equal>(tv0_2_4_param_0, tv0_2_4_param_1);
    TensorViewType expected_type{element::Bool::element_type(), Shape{2, 4}};
    EXPECT_EQ(*eq->get_value_type(), expected_type);
}

TEST(type_prop, binary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Add>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Operands for arithmetic operators must have numeric element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Negative>(tv0_2_4_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Operands for arithmetic operators must have numeric element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_deduce)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
    auto bc_vt = bc->get_value_type();
    ASSERT_EQ(*bc_vt, TensorViewType(element::Float32::element_type(), Shape{2, 4}));
}

TEST(type_prop, select_shape_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{3, 5}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments must have the same shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_b)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 5}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments must have the same shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_c)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 5}));
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments must have the same shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Argument 0 for arithmetic operators must have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_bc)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), Shape{2, 4}));
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Arguments 1 and 2 must have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_deduce)
{
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
    ASSERT_EQ(*(r0->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{4}));

    auto r1 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{1});
    ASSERT_EQ(*(r1->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{2}));

    auto r01 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 1});
    ASSERT_EQ(*(r01->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{}));

    auto r_none = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{});
    ASSERT_EQ(*(r_none->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{2, 4}));
}

TEST(type_prop, reduce_nonscalar)
{
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
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
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
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

TEST(type_prop, reduce_function_return_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Equal>(f_param_0, f_param_1), rt, op::Parameters{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg0_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Bool::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_0, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_2 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(
        f_param_0 + f_param_1 + f_param_2, rt, op::Parameters{f_param_0, f_param_1, f_param_2});

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
    auto param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4}));
    auto param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    try
    {
        auto r = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
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
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_f = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>((A + B * C), rt_f, op::Parameters{A, B, C});

    // Now make "f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Z = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto r = make_shared<op::FunctionCall>(f, Nodes{X, Y, Z});
    auto r_p_r = r + r;

    auto r_p_r_vt = r_p_r->get_value_type();
    ASSERT_EQ(*r_p_r_vt, TensorViewType(element::Float32::element_type(), shape));
}

TEST(type_prop, reshape_deduce_s2v)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1});
    ASSERT_EQ(*(r->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{}));
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1, 1});
    ASSERT_EQ(*(r->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_v2s)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{1}));
    auto r = make_shared<op::Reshape>(param, AxisVector{0}, Shape{});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, reshape_deduce_m2s)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{1, 1}));
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, reshape_deduce_t2s)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{1, 1, 1}));
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, reshape_deduce_m2v_01)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4}));
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{12});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{12}));
}

TEST(type_prop, reshape_deduce_m2v_10)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4}));
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{12});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{12}));
}

TEST(type_prop, reshape_deduce_t2v_012)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{60});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{60}));
}

TEST(type_prop, reshape_deduce_t2v_120)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{60});
    ASSERT_EQ(*(r->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{60}));
}

TEST(type_prop, reshape_deduce_not_enough_axes)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Not enough axes not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Input axis order for reshape is not a permutation of argument's axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_too_many_axes)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0, 3}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Input axis order for reshape is not a permutation of argument's axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_duplicate_axes)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Input axis order for reshape is not a permutation of argument's axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 4, 5}));
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{3, 3, 3});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Product of output shape dimensions does not match "
                              "product of argument shape dimensions for reshape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto sl = make_shared<op::Slice>(param, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(*(sl->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{3}));
}

TEST(type_prop, slice_deduce_matrix)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{3, 6}));
}

TEST(type_prop, slice_deduce_matrix_strided)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Shape{3, 2});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{1, 3}));
}

TEST(type_prop, slice_deduce_matrix_strided_uneven)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Shape{3, 4});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{1, 2}));
}

TEST(type_prop, slice_deduce_vector_edge)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(*(sl->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{6}));
}

TEST(type_prop, slice_deduce_matrix_edge)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, slice_deduce_matrix_zero_cols)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 0}));
}

TEST(type_prop, slice_deduce_matrix_zero_zero)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(*(sl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{0, 0}));
}

TEST(type_prop, slice_deduce_vector_invalid_strides)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7}, Shape{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string(
                      "Number of strides provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Upper bound for slice is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Upper bound for slice is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Lower bound for slice is greater than upper bound"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Lower bound for slice is greater than upper bound"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_missing)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of lower bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_missing)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of upper bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_extra)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of lower bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_extra)
{
    auto param = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of upper bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scalar_constant_deduce_float32)
{
    auto c = op::Constant::create(element::Float32::element_type(), Shape{}, {208});
    ASSERT_EQ(*(c->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool)
{
    auto c = op::Constant::create(element::Bool::element_type(), Shape{}, {1});
    ASSERT_EQ(*(c->get_value_type()), TensorViewType(element::Bool::element_type(), Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32)
{
    auto c =
        op::Constant::create(element::Float32::element_type(), Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(*(c->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool)
{
    auto c = op::Constant::create(element::Bool::element_type(), Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(*(c->get_value_type()), TensorViewType(element::Bool::element_type(), Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count)
{
    try
    {
        auto c = op::Constant::create(element::Bool::element_type(), Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Constant does not have the expected number of literals"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(*(rsl->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 6}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{1, 3}));
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Shape{3, 2});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided_uneven)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{1, 2}));
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Shape{3, 4});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_edge)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(*(rsl->get_value_type()), TensorViewType(element::Float32::element_type(), Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix_edge)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_cols)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 0}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_zero)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{0, 0}));
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(*(rsl->get_value_type()),
              TensorViewType(element::Float32::element_type(), Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_invalid_strides)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{4}));
    try
    {
        auto sl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{0}, Coordinate{7}, Shape{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string(
                      "Number of strides provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_rank_mismatch)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 6, 5}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Replace-slice argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_element_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), Shape{3, 6}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument element type mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for replace-slice arguments do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{3, 6}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{1, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Shape of replacement tensor does not match slice shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch_strided)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 6}));
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{1, 1}, Coordinate{5, 7}, Coordinate{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Shape of replacement tensor does not match slice shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{7}));
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Upper bound for slice is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 9}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Upper bound for slice is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{0}));
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Lower bound for slice is greater than upper bound"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 0}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Lower bound for slice is greater than upper bound"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_missing)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 6}));
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of lower bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_missing)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 6}));
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of upper bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_extra)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 6}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of lower bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_extra)
{
    auto param0 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 8}));
    auto param1 = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{6, 6}));
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string(
                "Number of upper bounds provided for slice does not match number of input axes"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_scalar)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{});
    auto oh = make_shared<op::OneHot>(param, Shape{9}, 0);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{9}));
}

TEST(type_prop, one_hot_deduce_vector_0)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{9, 8}, 0);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{9, 8}));
}

TEST(type_prop, one_hot_deduce_vector_1)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{8, 9}, 1);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{8, 9}));
}

TEST(type_prop, one_hot_deduce_matrix_0)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{2, 12, 24}, 0);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{2, 12, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_1)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 2, 24}, 1);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{12, 2, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_2)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 2}, 2);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Int32::element_type(), Shape{12, 24, 2}));
}

TEST(type_prop, one_hot_deduce_floating_point)
{
    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 2);
    auto oh_vt = oh->get_value_type();
    ASSERT_EQ(*oh_vt, TensorViewType(element::Float32::element_type(), Shape{12, 24, 8}));
}

TEST(type_prop, one_hot_deduce_axis_oob)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("One-hot axis is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_shape_incompatible)
{
    auto param = make_shared<op::Parameter>(element::Int32::element_type(), Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 22, 8}, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible one-hot output shape not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("One-hot argument shape is not compatible with desired output shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_1d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10});
    auto conv = make_shared<op::Convolution>(param0, param1);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 91}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), Shape{100});
    EXPECT_EQ(conv->get_output_image_shape(), Shape{91});

    EXPECT_EQ(conv->get_window_physical_shape(), Shape{10});
    EXPECT_EQ(conv->get_window_virtual_shape(), Shape{10});

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 1);
}

TEST(type_prop, conv_1d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 46}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), Shape{100});
    EXPECT_EQ(conv->get_output_image_shape(), Shape{46});

    EXPECT_EQ(conv->get_window_physical_shape(), Shape{10});
    EXPECT_EQ(conv->get_window_virtual_shape(), Shape{10});

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 1);
}

TEST(type_prop, conv_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 5});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), Shape{5});
    EXPECT_EQ(conv->get_output_image_shape(), Shape{2});

    EXPECT_EQ(conv->get_window_physical_shape(), Shape{2});
    EXPECT_EQ(conv->get_window_virtual_shape(), Shape{2});

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 1);
}

TEST(type_prop, conv_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 6});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 3}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), Shape{6});
    EXPECT_EQ(conv->get_output_image_shape(), Shape{3});

    EXPECT_EQ(conv->get_window_physical_shape(), Shape{2});
    EXPECT_EQ(conv->get_window_virtual_shape(), Shape{2});

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 1);
}

TEST(type_prop, conv_1d_deduce_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 82}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), Shape{100});
    EXPECT_EQ(conv->get_output_image_shape(), Shape{82});

    EXPECT_EQ(conv->get_window_physical_shape(), Shape{19});
    EXPECT_EQ(conv->get_window_virtual_shape(), Shape{10});

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 1);
}

TEST(type_prop, conv_2d_deduce)
{
    // Deduce type
    auto param0 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100, 150});
    auto param1 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10, 20});
    auto conv = make_shared<op::Convolution>(param0, param1);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), (Shape{100, 150}));
    EXPECT_EQ(conv->get_output_image_shape(), (Shape{91, 131}));

    EXPECT_EQ(conv->get_window_physical_shape(), (Shape{10, 20}));
    EXPECT_EQ(conv->get_window_virtual_shape(), (Shape{10, 20}));

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 2);
}

TEST(type_prop, conv_2d_deduce_strided)
{
    // Deduce type
    auto param0 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100, 150});
    auto param1 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 46, 44}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), (Shape{100, 150}));
    EXPECT_EQ(conv->get_output_image_shape(), (Shape{46, 44}));

    EXPECT_EQ(conv->get_window_physical_shape(), (Shape{10, 20}));
    EXPECT_EQ(conv->get_window_virtual_shape(), (Shape{10, 20}));

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 2);
}

TEST(type_prop, conv_2d_deduce_strided_dilated)
{
    // Deduce type
    auto param0 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 100, 150});
    auto param1 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 37, 38}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), (Shape{100, 150}));
    EXPECT_EQ(conv->get_output_image_shape(), (Shape{37, 38}));

    EXPECT_EQ(conv->get_window_physical_shape(), (Shape{28, 39}));
    EXPECT_EQ(conv->get_window_virtual_shape(), (Shape{10, 20}));

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 2);
}

TEST(type_prop, conv_2d_deduce_strided_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 7, 8});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 2, 3});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), (Shape{7, 8}));
    EXPECT_EQ(conv->get_output_image_shape(), (Shape{2, 2}));

    EXPECT_EQ(conv->get_window_physical_shape(), (Shape{4, 5}));
    EXPECT_EQ(conv->get_window_virtual_shape(), (Shape{2, 3}));

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 2);
}

TEST(type_prop, conv_3d_deduce_strided_dilated_small)
{
    // Deduce type
    auto param0 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{64, 3, 7, 8, 10});
    auto param1 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    auto conv_vt = conv->get_value_type();
    EXPECT_EQ(*conv_vt, TensorViewType(element::Float32::element_type(), Shape{64, 128, 2, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));

    EXPECT_EQ(conv->get_input_channel_count(), 3);
    EXPECT_EQ(conv->get_output_channel_count(), 128);

    EXPECT_EQ(conv->get_input_image_shape(), (Shape{7, 8, 10}));
    EXPECT_EQ(conv->get_output_image_shape(), (Shape{2, 2, 2}));

    EXPECT_EQ(conv->get_window_physical_shape(), (Shape{4, 5, 3}));
    EXPECT_EQ(conv->get_window_virtual_shape(), (Shape{2, 3, 2}));

    EXPECT_EQ(conv->get_batch_size(), 64);
    EXPECT_EQ(conv->get_image_dimension_count(), 3);
}

TEST(type_prop, conv_invalid_0d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution image batch input must have rank of at "
                              "least 3 (one batch axis, one input-channel axis, at "
                              "least one image dimension)."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_1d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution image batch input must have rank of at "
                              "least 3 (one batch axis, one input-channel axis, at "
                              "least one image dimension)."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_2d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 6});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 6});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution image batch input must have rank of at "
                              "least 3 (one batch axis, one input-channel axis, at "
                              "least one image dimension)."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_batch_size)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{0, 6, 1});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{0, 6, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution image batch size is zero."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_input_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 0, 1});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{5, 0, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 input channels not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution requires at least one input channel."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_many)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 =
        make_shared<op::Parameter>(element::Float32::element_type(), Shape{5, 2, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many filter dimensions not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Convolution filter input must have rank of 2 + n_image_dimensions."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_few)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{5, 2, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few filter dimensions not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Convolution filter input must have rank of 2 + n_image_dimensions."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_output_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{0, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 output channels not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution requires at least one output channel."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_channel_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with channel count mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Convolution image batch and filter input channel counts do not match."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution window movement stride rank does not "
                              "match number of image dimensions."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong dilation stride rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution window dilation stride rank does not "
                              "match number of image dimensions."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_image_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 0, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length image axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution input image dimension is zero."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 0});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution window shape has a zero-length axis."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length dilation stride axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution window axis dilation stride is zero."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilated_window_too_large)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 8, 8});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{1, 1}, Strides{4, 4});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized dilated window not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Convolution window after dilation is larger than the image."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{0, 1});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length movement stride axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Convolution window axis movement stride is zero."));
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
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{}));
}

TEST(type_prop, reverse_1d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5}));
}

TEST(type_prop, reverse_1d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5}));
}

TEST(type_prop, reverse_2d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6}));
}

TEST(type_prop, reverse_3d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_2)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{2});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_02)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 2});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_12)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1, 2});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_012)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1, 2});
    auto rev_vt = rev->get_value_type();
    EXPECT_EQ(*rev_vt, TensorViewType(element::f32, Shape{5, 6, 7}));
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
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reverse axis 3 is out of bounds (input rank is 3)."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_1d)
{
    auto param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{16}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{1};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(*(rw->get_value_type()), TensorViewType(element::f32, Shape{13}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_even)
{
    auto param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{16}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(*(rw->get_value_type()), TensorViewType(element::f32, Shape{4}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(*(rw->get_value_type()), TensorViewType(element::f32, Shape{4}));
}

TEST(type_prop, reduce_window_deduce_2d_strided_uneven)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4, 2};
    Strides move_strides{4, 3};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(*(rw->get_value_type()), TensorViewType(element::f32, Shape{4, 3}));
}

TEST(type_prop, reduce_window_deduce_3d_strided_uneven)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(*(rw->get_value_type()), TensorViewType(element::f32, Shape{4, 3, 6}));
}

TEST(type_prop, reduce_window_deduce_non_scalar_init)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{3}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::i32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_2 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(
        f_param_0 + f_param_1, rt, op::Parameters{f_param_0, f_param_1, f_param_2});

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

TEST(type_prop, reduce_window_deduce_param_0_wrong_type)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::i32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, rt, op::Parameters{f_param_0, f_param_1});

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
        EXPECT_EQ(error.what(), std::string("Parameter 0 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_1_wrong_type)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::i32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0, rt, op::Parameters{f_param_0, f_param_1});

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
        EXPECT_EQ(error.what(), std::string("Parameter 1 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_multi_output)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::f32, Shape{});
    auto rts = std::vector<std::shared_ptr<const ValueType>>{rt, rt};
    auto f = make_shared<Function>(Nodes{f_param_0 + f_param_1, f_param_0 * f_param_1},
                                   rts,
                                   op::Parameters{f_param_0, f_param_1});

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

TEST(type_prop, reduce_window_reduction_function_return_type_mismatch)
{
    auto param_0 =
        make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{18, 10, 15}));
    auto param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));

    auto f_param_0 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto f_param_1 = make_shared<op::Parameter>(make_shared<TensorViewType>(element::f32, Shape{}));
    auto rt = make_shared<TensorViewType>(element::i32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Convert>(f_param_0 + f_param_1, element::i32),
                                   rt,
                                   op::Parameters{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Reduction function return type mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
