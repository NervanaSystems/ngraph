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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/node.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

TEST(eval, max_eval_parameter)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{});

    auto result = maximum_value(p);
    EXPECT_FALSE(result.first);
    EXPECT_EQ(result.second, 0);
}

TEST(eval, max_eval_constant)
{
    auto c = op::Constant::create<int64_t>(element::i64, Shape{}, {27});
    auto result = maximum_value(c);
    ASSERT_TRUE(result.first);
    EXPECT_EQ(result.second, 27);
}

TEST(eval, max_eval_minimum_constant)
{
    auto c = op::Constant::create<int64_t>(element::i64, Shape{}, {27});
    auto p = make_shared<op::Parameter>(element::i64, Shape{});
    auto m = make_shared<op::Minimum>(c, p);
    auto result = maximum_value(m);
    ASSERT_TRUE(result.first);
    EXPECT_EQ(result.second, 27);
}

TEST(eval, evaluate_shape_of)
{
    auto p = make_shared<op::Parameter>(element::f32, PartialShape{-1, -1});
    auto so = make_shared<op::v0::ShapeOf>(p);

    auto p_arg = op::Constant::create<float>(
        element::f32, Shape{2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto p_arg_tensor = op::v0::Constant::create_evaluator_tensor(p_arg);
    map<RawNodeOutput, EvaluatorTensorPtr> value_map;
    value_map[p->output(0)] = p_arg_tensor;
    evaluate_nodes(value_map, {so->output(0)});
    auto value = value_map.find(so->output(0));
    ASSERT_TRUE(value != value_map.end());
    auto c = value->second;
    ASSERT_TRUE(c);
    EXPECT_EQ(c->get_element_type(), element::i64);
    EXPECT_EQ(c->get_partial_shape(), (PartialShape{2}));
    int64_t* shape = c->get_ptr<element::Type_t::i64>();
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
}
