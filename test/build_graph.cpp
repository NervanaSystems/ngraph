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


template<typename T, typename ...A>
std::shared_ptr<T> myfun(A&&... args)
{
    return std::make_shared<T>(args...);
}

template<>
std::shared_ptr<Parameter> myfun<Parameter> (ngraph::element::Type&& element_type, Shape&& shape)
{
    return make_shared<Parameter>(make_shared<TensorViewType>(element_type, shape));
}

TEST(build_graph, build_simple)
{
    // Function with 4 parameters
    auto arg0        = op::parameter(element::Float::element_type(), Shape{7, 3});
    auto arg1        = op::parameter(element::Float::element_type(), Shape{3});
    auto arg2        = op::parameter(element::Float::element_type(), Shape{32, 7});
    auto arg3        = op::parameter(element::Float::element_type(), Shape{32, 7});
    auto broadcast_1 = op::broadcast(arg3, Shape{10, 32, 7}, BroadcastOp::Axes{0});
    auto b1 = myfun<BroadcastOp>(arg3, Shape{10, 32, 7}, BroadcastOp::Axes{0});
    auto dot         = op::dot(arg2, arg0);
    ASSERT_EQ(dot->arguments()[0], arg2);
    ASSERT_EQ(dot->arguments()[1], arg0);

    auto cluster_0 = op::function(dot, {arg0, arg1, arg2, arg3});

    ASSERT_EQ(cluster_0->result(), dot);
}

// Check upcasting from ValueType.
TEST(build_graph, as_type)
{
    // Check upcasting a ValueType::ptr that is a TensorViewType to a TensorViewType and Tuple.
    ValueType::ptr tv_vt = make_shared<TensorViewType>(element::Float::element_type(), Shape{2, 3, 5});
    auto           tv_tv = dynamic_pointer_cast<TensorViewType>(tv_vt);
    ASSERT_EQ(tv_vt, tv_tv);
    auto tv_tp = dynamic_pointer_cast<TupleType>(tv_vt);
    ASSERT_EQ(nullptr, tv_tp);

    // Check upcasting a ValueType::ptr that is a TupleType to a TensorViewType and Tuple.
    ValueType::ptr tp_vt = make_shared<TupleType>(vector<ValueType::ptr>{tv_vt, tv_vt});
    auto           tp_tv = dynamic_pointer_cast<TensorViewType>(tp_vt);
    ASSERT_EQ(nullptr, tp_tv);
    auto tp_tp = dynamic_pointer_cast<TupleType>(tp_vt);
    ASSERT_EQ(tp_vt, tp_tp);
}

// Check node comparisons
TEST(build_graph, node_comparison)
{
    auto arg0 = op::parameter(element::Float::element_type(), {32, 3});
    auto arg1 = op::parameter(element::Float::element_type(), {3});
    auto arg2 = op::parameter(element::Float::element_type(), {32});

    auto dot = op::dot(arg0, arg1);
    auto add = op::add(dot, arg2);

    auto parg        = op::parameter(element::Float::element_type(), {});
    auto pattern_dot = op::dot(parg, parg);
    ASSERT_TRUE(pattern_dot->is_same_op_type(dot));
    // TODO This passes because typeid is not behaving as documented.
    // Need to figure out what's wrong.
    ASSERT_FALSE(pattern_dot->is_same_op_type(add));
}

TEST(build_graph, literal)
{
    // float scalar from a float
    auto float0 = FloatScalarConstant::make(3.0);
    auto float_scalar_type =  make_shared<TensorViewType>(element::Float::element_type(), Shape{});
    ASSERT_EQ(float0->value(), 3.0);
    ASSERT_EQ(*float0->type(), float_scalar_type);
    auto d = op::dot(float0, float0);
    ASSERT_EQ(d->arguments().at(0), float0);
    ASSERT_EQ(d->arguments().at(1), float0);

    // float scalar from an int
    auto float1 = FloatScalarConstant::make(3);
    ASSERT_EQ(float1->value(), 3);
    ASSERT_EQ(*float1->type(), float_scalar_type);
    
    auto int32_0 = Int32ScalarConstant::make(3.0);
    auto int32_scalar_type =  make_shared<TensorViewType>(element::Int32::element_type(), Shape{});
    ASSERT_EQ(int32_0->value(), 3);
    ASSERT_EQ(*int32_0->type(), int32_scalar_type);
    ASSERT_NE(*int32_0->type(), float_scalar_type);
}

// Check argument inverses
TEST(build_graph, arg_inverse)
{
}
