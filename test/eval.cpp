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
    auto p_arg = op::Constant::create<float>(
        element::f32, Shape{2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    HostTensorVector inputs;
    inputs.push_back(make_shared<HostTensor>(p_arg->output(0)));
    HostTensorVector outputs;
    auto result = fun->get_results()[0];
    auto result_tensor = make_shared<HostTensor>(result->output(0));
    outputs.push_back(result_tensor);
    ASSERT_TRUE(fun->evaluate(outputs, inputs));
    auto c = result_tensor;
    ASSERT_TRUE(c);
    EXPECT_EQ(c->get_element_type(), element::i64);
    EXPECT_EQ(c->get_partial_shape(), (PartialShape{2}));
    auto cshape = read_vector<int64_t>(c);
    vector<int64_t> arg_shape{2, 3};
    ASSERT_EQ(cshape, arg_shape);
}
