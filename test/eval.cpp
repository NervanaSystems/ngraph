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
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

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
    auto fun = make_shared<Function>(OutputVector{so}, ParameterVector{p});
    auto p_arg = op::Constant::create<float>(
        element::f32, Shape{2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    EvaluatorTensorVector inputs;
    inputs.push_back(runtime::HostTensor::create_evaluator_tensor(p_arg->get_output_element_type(0),
                                                                  p_arg->get_output_shape(0)));
    EvaluatorTensorVector outputs;
    auto result = fun->get_results()[0];
    auto result_tensor = runtime::HostTensor::create_evaluator_tensor(
        result->get_output_element_type(0), result->get_output_shape(0));
    outputs.push_back(result_tensor);
    ASSERT_TRUE(fun->evaluate(outputs, inputs));
    auto c = result_tensor->get_host_tensor();
    ASSERT_TRUE(c);
    EXPECT_EQ(c->get_element_type(), element::i64);
    EXPECT_EQ(c->get_partial_shape(), (PartialShape{2}));
    auto cshape = read_vector<int64_t>(c);
    vector<int64_t> arg_shape{2, 3};
    ASSERT_EQ(cshape, arg_shape);
}

TEST(eval, evaluate_dynamic_range_sum)
{
    auto p_start = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_stop = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_step = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto range = make_shared<op::v0::Range>(p_start, p_stop, p_step);
    auto add = make_shared<op::v1::Add>(range, p1);
    auto fun =
        make_shared<Function>(OutputVector{add}, ParameterVector{p_start, p_stop, p_step, p1});
    auto p_start_val = op::Constant::create<float>(element::f32, Shape{}, {1.0f});
    auto p_stop_val = op::Constant::create<float>(element::f32, Shape{}, {10.0f});
    auto p_step_val = op::Constant::create<float>(element::f32, Shape{}, {3.0f});
    auto p1_val = op::Constant::create<float>(element::f32, Shape{}, {7.0f});
    EvaluatorTensorVector inputs;
    inputs.push_back(op::v0::Constant::create_evaluator_tensor(p_start_val));
    inputs.push_back(op::v0::Constant::create_evaluator_tensor(p_stop_val));
    inputs.push_back(op::v0::Constant::create_evaluator_tensor(p_step_val));
    inputs.push_back(op::v0::Constant::create_evaluator_tensor(p1_val));
    EvaluatorTensorVector outputs;
    auto result = fun->get_results()[0];
    auto result_tensor =
        op::v0::Constant::create_evaluator_tensor(element::dynamic, PartialShape::dynamic());
    outputs.push_back(result_tensor);
    ASSERT_TRUE(fun->evaluate(outputs, inputs));
    auto c = result_tensor->get_constant();
    ASSERT_TRUE(c);
    EXPECT_EQ(c->get_output_element_type(0), element::f32);
    EXPECT_EQ(c->get_output_partial_shape(0), (PartialShape{3}));
    auto cval = c->get_vector<float>();
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(cval, seq);
}

TEST(eval, interpret_dynamic_range_sum)
{
    auto p_start = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_stop = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_step = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto range = make_shared<op::v0::Range>(p_start, p_stop, p_step);
    auto add = make_shared<op::v1::Add>(range, p1);
    auto fun =
        make_shared<Function>(OutputVector{add}, ParameterVector{p_start, p_stop, p_step, p1});
    auto backend = runtime::Backend::create("INTERPRETER");
    auto p_start_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_start_val, vector<float>{1.0f});
    auto p_stop_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_stop_val, vector<float>{10.0f});
    auto p_step_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_step_val, vector<float>{3.0f});
    auto p1_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p1_val, vector<float>{7.0f});
    vector<shared_ptr<runtime::Tensor>> results;
    // Interpreter provides the tensor
    results.push_back(nullptr);
    auto cfun = backend->compile(fun);
    cfun->call_dynamic({results}, {p_start_val, p_stop_val, p_step_val, p1_val});
    auto c = results[0];
    ASSERT_TRUE(c);
    EXPECT_EQ(c->get_element_type(), element::f32);
    EXPECT_EQ(c->get_partial_shape(), (PartialShape{3}));
    auto cval = read_vector<float>(c);
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(cval, seq);
}
