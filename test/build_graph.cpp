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

TEST(build_graph, build_simple)
{
    // Function with 4 parameters
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{7, 3});
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{3});
    auto arg2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{32, 7});
    auto arg3 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{32, 7});
    auto broadcast_1 = make_shared<op::Broadcast>(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = make_shared<op::Broadcast>(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::Dot>(arg2, arg0);
    ASSERT_EQ(dot->get_arguments()[0], arg2);
    ASSERT_EQ(dot->get_arguments()[1], arg0);

    auto result_type =
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{10, 32, 7});
    auto cluster_0 =
        make_shared<Function>(dot, result_type, op::Parameters{arg0, arg1, arg2, arg3});

    ASSERT_EQ(cluster_0->get_result(), dot);
}

// Check upcasting from ValueType.
TEST(build_graph, as_type)
{
    // Check upcasting a ValueType::ptr that is a TensorViewType to a TensorViewType and Tuple.
    auto tv_vt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 3, 5});
    auto tv_tv = dynamic_pointer_cast<TensorViewType>(tv_vt);
    ASSERT_EQ(tv_vt, tv_tv);
    auto tv_tp = dynamic_pointer_cast<TupleType>(tv_vt);
    ASSERT_EQ(nullptr, tv_tp);

    // Check upcasting a ValueType::ptr that is a TupleType to a TensorViewType and Tuple.
    auto tp_vt = make_shared<TupleType>(ValueTypes{tv_vt, tv_vt});
    auto tp_tv = dynamic_pointer_cast<TensorViewType>(tp_vt);
    ASSERT_EQ(nullptr, tp_tv);
    auto tp_tp = dynamic_pointer_cast<TupleType>(tp_vt);
    ASSERT_EQ(tp_vt, tp_tp);
}

// Check node comparisons
TEST(build_graph, node_comparison)
{
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{32, 3});
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{3});
    auto arg2 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{32});

    auto dot = make_shared<op::Dot>(arg0, arg1);
    auto add = make_shared<op::Add>(dot, arg2);

    auto parg = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto pattern_dot = make_shared<op::Dot>(parg, parg);
    ASSERT_TRUE(pattern_dot->is_same_op_type(dot));
    // TODO This passes because typeid is not behaving as documented.
    // Need to figure out what's wrong.
    ASSERT_FALSE(pattern_dot->is_same_op_type(add));
}

TEST(build_graph, literal)
{
    // float scalar from a float
    //auto float0 = FloatScalarConstant::make(3.0);
    auto float0 = make_shared<op::Float32ScalarConstant>(3.0);
    auto float_scalar_type = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    ASSERT_EQ(float0->get_value(), 3.0);
    ASSERT_EQ(*float0->get_value_type(), *float_scalar_type);
    auto d = make_shared<op::Dot>(float0, float0);
    ASSERT_EQ(d->get_arguments().at(0), float0);
    ASSERT_EQ(d->get_arguments().at(1), float0);

    // float scalar from an int
    auto float1 = make_shared<op::Float32ScalarConstant>(3);
    ASSERT_EQ(float1->get_value(), 3);
    ASSERT_EQ(*float1->get_value_type(), *float_scalar_type);

    auto int32_0 = make_shared<op::Int32ScalarConstant>(3.0);
    auto int32_scalar_type = make_shared<TensorViewType>(element::Int32::element_type(), Shape{});
    ASSERT_EQ(int32_0->get_value(), 3);
    ASSERT_EQ(*int32_0->get_value_type(), *int32_scalar_type);
    ASSERT_NE(*int32_0->get_value_type(), *float_scalar_type);
}

TEST(build_graph, tensor)
{
    // float scalar from a float
    //auto float0 = FloatScalarConstant::make(3.0);
    auto float0 = make_shared<op::Float32TensorConstant>(Shape{2, 3});
    auto float_tensor_type =
        make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 3});
    ASSERT_EQ(*float0->get_value_type(), *float_tensor_type);
    auto d = make_shared<op::Dot>(float0, float0);
    ASSERT_EQ(d->get_arguments().at(0), float0);
    ASSERT_EQ(d->get_arguments().at(1), float0);

    auto int32_0 = make_shared<op::Int32TensorConstant>(Shape{3, 5});
    auto int32_tensor_type =
        make_shared<TensorViewType>(element::Int32::element_type(), Shape{3, 5});
    ASSERT_EQ(*int32_0->get_value_type(), *int32_tensor_type);
    ASSERT_NE(*int32_0->get_value_type(), *float_tensor_type);
}

TEST(build_graph, set_value_type_checked)
{
    auto untyped_param = make_shared<op::Parameter>();
    try
    {
        untyped_param->set_value_type_checked(
            make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 4}));
    }
    catch (...)
    {
        FAIL() << "Setting value type for first time type failed.";
    }
    try
    {
        untyped_param->set_value_type_checked(
            make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 4}));
    }
    catch (...)
    {
        FAIL() << "Setting value type to same type failed.";
    }
    try
    {
        untyped_param->set_value_type_checked(
            make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 5}));
        FAIL() << "Setting value type to a different shape did not fail.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Setting value type to a different ValueType"));
    }
    catch (...)
    {
        FAIL() << "Setting value type to a different shape did not failed with incorrect error.";
    }
    try
    {
        untyped_param->set_value_type_checked(
            make_shared<TensorViewType>(element::Int32::element_type(), Shape{4, 4}));
        FAIL() << "Setting value type to a different element type did not fail.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Setting value type to a different ValueType"));
    }
    catch (...)
    {
        FAIL() << "Setting value type to a different element type did not failed with incorrect "
                  "error.";
    }

    auto param = make_shared<op::Parameter>(element::Float32::element_type(), Shape{4, 4});
    try
    {
        param->set_value_type_checked(
            make_shared<TensorViewType>(element::Float32::element_type(), Shape{4, 4}));
    }
    catch (...)
    {
        FAIL() << "Setting value type to same type failed.";
    }
}

// Check argument inverses
TEST(build_graph, arg_inverse)
{
}
