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
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/transpose.hpp"
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

#ifdef NGRAPH_INTERPRETER_ENABLE
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
    auto result = backend->create_tensor();
    auto cfun = backend->compile(fun);
    cfun->call({result}, {p_start_val, p_stop_val, p_step_val, p1_val});
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{3}));
    auto result_val = read_vector<float>(result);
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(result_val, seq);
}
#endif

TEST(eval, evaluate_dynamic_concat)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto concat = make_shared<op::v0::Concat>(NodeVector{arg1, arg2}, 1);
    auto fun = make_shared<Function>(OutputVector{concat}, ParameterVector{arg1, arg2});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({1, 1}, {1.0f}),
                               make_host_tensor<element::Type_t::f32>({1, 2}, {8.0f, 10.0f})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{1, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1.0f, 8.0f, 10.0f};
    ASSERT_EQ(cval, out);
}

template <element::Type_t T>
void test_eval(shared_ptr<Function> fun,
               vector<vector<float>>& inputs,
               vector<Shape>& x_shapes,
               vector<Shape>& result_shapes,
               vector<vector<float>>& results)
{
    using IN_T = typename element_type_traits<T>::value_type;
    std::vector<std::vector<IN_T>> perms{{0, 1}, {1, 0}, {2, 1, 0}};
    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        // If I have the result_tensor defined outside the loop, we fail
        // dynamic shape vs real shape check for second test case onwards.
        // Please confirm that we are NOT allowed to reuse result tensors! TBD in code review
        auto result_tensor = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result_tensor},
                                  {make_host_tensor<element::Type_t::f32>(x_shapes[i], inputs[i]),
                                   make_host_tensor<T>(Shape{perms[i].size()}, perms[i])}));

        ASSERT_EQ(result_tensor->get_shape(), result_shapes[i]);
        auto actual_results = read_vector<float>(result_tensor);
        ASSERT_EQ(actual_results, results[i]);
    }
}
TEST(eval, eval_transpose)
{
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<shared_ptr<op::Parameter>> axes;
    axes.push_back(make_shared<op::Parameter>(element::i8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()}));

    axes.push_back(make_shared<op::Parameter>(element::u8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u64, PartialShape{Dimension::dynamic()}));

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};

    std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> result_shapes{Shape{2, 3}, Shape{3, 2}, {3, 2, 2}};
    std::vector<std::vector<float>> results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (auto& axis : axes)
    {
        auto x_transpose = make_shared<op::v1::Transpose>(x, axis);
        auto fun = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{x, axis});

        {
            switch (axis->get_element_type())
            {
            case element::Type_t::i8:
                test_eval<element::Type_t::i8>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::i16:
                test_eval<element::Type_t::i16>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::i32:
                test_eval<element::Type_t::i32>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::i64:
                test_eval<element::Type_t::i64>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::u8:
                test_eval<element::Type_t::u8>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::u16:
                test_eval<element::Type_t::u16>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::u32:
                test_eval<element::Type_t::u32>(fun, inputs, x_shapes, result_shapes, results);
                break;
            case element::Type_t::u64:
                test_eval<element::Type_t::u64>(fun, inputs, x_shapes, result_shapes, results);
                break;
            default: NGRAPH_CHECK(false, "Invalid type"); break;
            }
        }
    }
}
