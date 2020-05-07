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

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/node.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/round.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(eval, bad_get_data_ptr)
{
    HostTensor c(element::f32, Shape{});
    *c.get_data_ptr<float>() = 1.0;
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::f32>(), 1.0);
    try
    {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try
    {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}

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
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2}));
    auto result_shape = read_vector<int64_t>(result);
    vector<int64_t> arg_shape{2, 3};
    ASSERT_EQ(result_shape, arg_shape);
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
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({}, {1.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {10.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {3.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {7.0f})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3}));
    auto cval = read_vector<float>(result_tensor);
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
    auto result = make_shared<HostTensor>();
    auto cfun = backend->compile(fun);
    cfun->call({result}, {p_start_val, p_stop_val, p_step_val, p1_val});
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{3}));
    auto result_val = read_vector<float>(result);
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(result_val, seq);
}

TEST(eval, evaluate_abs)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto abs = make_shared<op::Abs>(p);
    auto fun = make_shared<Function>(OutputVector{abs}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_erf)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto erf = make_shared<op::Erf>(p);
    auto fun = make_shared<Function>(OutputVector{erf}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::erf(0.0f),
                        std::erf(-1.0f),
                        std::erf(-2.0f),
                        std::erf(-3.0f),
                        std::erf(4.0f),
                        std::erf(5.0f)};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_exp)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto exp = make_shared<op::Exp>(p);
    auto fun = make_shared<Function>(OutputVector{exp}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::exp(0.0f),
                        std::exp(-1.0f),
                        std::exp(-2.0f),
                        std::exp(-3.0f),
                        std::exp(4.0f),
                        std::exp(5.0f)};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_floor)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto floor = make_shared<op::Floor>(p);
    auto fun = make_shared<Function>(OutputVector{floor}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{2, 2}, {-2.5f, -2.0f, 0.3f, 4.8f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-3.0f, -2.0f, 0.0f, 4.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_floor_int32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 2});
    auto floor = make_shared<op::Floor>(p);
    auto fun = make_shared<Function>(OutputVector{floor}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 2}, {-2, -136314888, 0x40000010, 0x40000001})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{-2, -136314888, 0x40000010, 0x40000001};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_log)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    auto log = make_shared<op::Log>(p);
    auto fun = make_shared<Function>(OutputVector{log}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(
                          Shape{2, 2, 2}, {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::log(0.125f),
                        std::log(0.25f),
                        std::log(0.5f),
                        std::log(1.f),
                        std::log(2.f),
                        std::log(4.f),
                        std::log(8.f),
                        std::log(16.f)};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_negative_f32)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto negate = make_shared<op::Negative>(p);
    auto fun = make_shared<Function>(OutputVector{negate}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(
            Shape{2, 5},
            {1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_negative_i32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    auto negate = make_shared<op::Negative>(p);
    auto fun = make_shared<Function>(OutputVector{negate}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -2, 1, 8, -8, 17, 0})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{-1, -8, 8, -17, 2, -1, -8, 8, -17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_relu_2Ffprop_f32)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto relu = make_shared<op::Relu>(p);
    auto fun = make_shared<Function>(OutputVector{relu}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -0.5, 0.1, 8.5, -8, 17, -0.5})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 8, 0, 17, 0, 0.1, 8.5, 0, 17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_relu_2Ffprop_i32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    auto relu = make_shared<op::Relu>(p);
    auto fun = make_shared<Function>(OutputVector{relu}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -2, 1, 8, -8, 17, -1})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_round)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{5});
    auto round = make_shared<op::Round>(p);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1.0f, 2.0f, 2.0f, 2.0f, -4.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_round_2D)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    auto round = make_shared<op::Round>(p);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3, 5},
                                                                      {0.1f,
                                                                       0.5f,
                                                                       0.9f,
                                                                       1.2f,
                                                                       1.5f,
                                                                       1.8f,
                                                                       2.3f,
                                                                       2.5f,
                                                                       2.7f,
                                                                       -1.1f,
                                                                       -1.5f,
                                                                       -1.9f,
                                                                       -2.2f,
                                                                       -2.5f,
                                                                       -2.8f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sigmoid)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto sigmoid = make_shared<op::Sigmoid>(p);
    auto fun = make_shared<Function>(OutputVector{sigmoid}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    float x1 = 1.0f;
    float x2 = 4.0f;
    float sigma1 = 1.0f / (1.0f + std::exp(-x1));
    float sigma2 = 1.0f / (1.0f + std::exp(-x2));
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{1, 1, 2, 2}, {x1, x2, x1, x2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{sigma1, sigma2, sigma1, sigma2};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sign)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto sign = make_shared<op::Sign>(p);
    auto fun = make_shared<Function>(OutputVector{sign}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{2, 3}, {1, -2, 0, -4.8f, 4.8f, -0.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, -1, 0, -1, 1, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sin)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto sin = make_shared<op::Sin>(p);
    auto fun = make_shared<Function>(OutputVector{sin}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(
            Shape{11}, {0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{0.00000000f,
                        0.24740396f,
                        -0.24740396f,
                        0.47942554f,
                        -0.47942554f,
                        0.84147098f,
                        -0.84147098f,
                        0.90929743f,
                        -0.90929743f,
                        -0.75680250f,
                        0.75680250f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sinh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sinh = make_shared<op::Sinh>(p);
    auto fun = make_shared<Function>(OutputVector{sinh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });
    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_sqrt)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sqrt = make_shared<op::Sqrt>(p);
    auto fun = make_shared<Function>(OutputVector{sqrt}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{16, 4, 81, 100, 10000, 0};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{4, 2, 9, 10, 100, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_acos)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto acos = make_shared<op::Acos>(p);
    auto fun = make_shared<Function>(OutputVector{acos}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::acos(x); });
    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_asin)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto asin = make_shared<op::Asin>(p);
    auto fun = make_shared<Function>(OutputVector{asin}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::asin(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_atan)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto atan = make_shared<op::Atan>(p);
    auto fun = make_shared<Function>(OutputVector{atan}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::atan(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_ceiling)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto ceil = make_shared<op::Ceiling>(p);
    auto fun = make_shared<Function>(OutputVector{ceil}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-2.5f, -2.0f, 0.3f, 4.8f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{2, 2}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-2.0f, -2.0f, 1.0f, 5.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_cos)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto cos = make_shared<op::Cos>(p);
    auto fun = make_shared<Function>(OutputVector{cos}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::cos(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_cosh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto cosh = make_shared<op::Cosh>(p);
    auto fun = make_shared<Function>(OutputVector{cosh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::cosh(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_tan)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto tan = make_shared<op::Tan>(p);
    auto fun = make_shared<Function>(OutputVector{tan}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::tan(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_tanh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto tanh = make_shared<op::Tanh>(p);
    auto fun = make_shared<Function>(OutputVector{tanh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::tanh(x); });

    ASSERT_EQ(result_val, input);
}

TEST(eval, evaluate_not)
{
    auto p = make_shared<op::Parameter>(element::boolean, Shape{2, 2});
    auto op_not = make_shared<op::Not>(p);
    auto fun = make_shared<Function>(OutputVector{op_not}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::boolean>(Shape{2, 2}, {1, 0, 1, 0})}));
    EXPECT_EQ(result->get_element_type(), element::boolean);
    auto result_val = read_vector<char>(result);
    vector<char> expec{0, 1, 0, 1};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_logical_not)
{
    auto p = make_shared<op::Parameter>(element::boolean, Shape{2, 2});
    auto logical_not = make_shared<op::v1::LogicalNot>(p);
    auto fun = make_shared<Function>(OutputVector{logical_not}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::boolean>(Shape{2, 2}, {1, 0, 1, 0})}));
    EXPECT_EQ(result->get_element_type(), element::boolean);
    auto result_val = read_vector<char>(result);
    vector<char> expec{0, 1, 0, 1};
    ASSERT_EQ(result_val, expec);
}
