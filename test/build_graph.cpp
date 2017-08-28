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

using namespace std;
using namespace ngraph;

TEST(build_graph, build_simple)
{
    // Function with 4 parameters
    auto cluster_0 = make_shared<Function>(4);
    cluster_0->result()->type(element::float32_t, {32, 3});
    cluster_0->parameter(0)->type(element::float32_t, {7, 3});
    cluster_0->parameter(1)->type(element::float32_t, {3});
    cluster_0->parameter(2)->type(element::float32_t, {32, 7});
    cluster_0->parameter(3)->type(element::float32_t, {32, 7});
    auto arg3 = cluster_0->parameter(3);
    // call broadcast op on arg3, broadcasting on axis 0.
    auto broadcast_1 = op::broadcast(arg3, {10, 32, 7}, {0});
    auto arg2        = cluster_0->parameter(2);
    auto arg0        = cluster_0->parameter(0);
    // call dot op
    auto dot = op::dot(arg2, arg0);
    ASSERT_EQ(dot->arguments()[0], arg2);
    ASSERT_EQ(dot->arguments()[1], arg0);
    // Function returns tuple of dot and broadcast_1.
    cluster_0->result()->value(dot);

    ASSERT_EQ(cluster_0->result()->value(), dot);
}

// Check upcasting from ValueType.
TEST(build_graph, as_type)
{
    // Check upcasting a ValueType::ptr that is a TensorViewType to a TensorViewType and Tuple.
    ValueType::ptr tv_vt = make_shared<TensorViewType>(element::float32_t, Shape{2, 3, 5});
    TensorViewType* tv_tv = tv_vt->as<TensorViewType*>();
    ASSERT_EQ(tv_vt.get(), tv_tv);
    TupleType* tv_tp = tv_vt->as<TupleType*>();
    ASSERT_EQ(nullptr, tv_tp);

    // Check upcasting a ValueType::ptr that is a TupleType to a TensorViewType and Tuple.
    ValueType::ptr tp_vt = make_shared<TupleType>(vector<ValueType::ptr>{tv_vt, tv_vt});
    TensorViewType* tp_tv = tp_vt->as<TensorViewType*>();
    ASSERT_EQ(nullptr, tp_tv);
    TupleType* tp_tp = tp_vt->as<TupleType*>();
    ASSERT_EQ(tp_vt.get(), tp_tp);
}
