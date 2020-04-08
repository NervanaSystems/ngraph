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

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(nop_elimination, eliminate_pad)
{
    Shape shape_a{2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    CoordinateDiff padding_below{0};
    CoordinateDiff padding_above{0};
    auto p = make_shared<op::Pad>(A, B, padding_below, padding_above);
    auto f = make_shared<Function>(make_shared<op::Abs>(p), ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Pad>(f), 0);
}

TEST(nop_elimination, eliminate_sum)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::Sum>(A, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Sum>(f), 0);
}

TEST(nop_elimination, eliminate_convert)
{
    Shape shape{};
    auto type = element::f32;
    auto A = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Convert>(A, element::f32);
    auto f = make_shared<Function>(make_shared<op::Abs>(c), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Convert>(f), 0);
}

TEST(nop_elimination, eliminate_slice)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 2});
    auto f = make_shared<Function>(make_shared<op::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 0);
}

TEST(nop_elimination, eliminate_broadcast)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Broadcast>(A, shape, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::Abs>(b), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
}

TEST(nop_elimination, eliminate_stop_gradient)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::StopGradient>(A), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::StopGradient>(f), 0);
}

TEST(nop_elimination, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::NopElimination>();
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(nop_elimination, reshape_elimination_v1)
{
    auto generate_func = [](bool zero) {
        auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape{8, 16, 2, 3});
        auto pattern = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{8, 16, 2, 3});
        auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, zero);
        auto abs = std::make_shared<op::Abs>(reshape_v1);
        return std::make_shared<Function>(NodeVector{abs}, ParameterVector{arg});
    };

    auto func = generate_func(false);
    auto nopass_func = generate_func(false);
    auto func_zero = generate_func(true);
    auto nopass_func_zero = generate_func(true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(func);
    pass_manager.run_passes(func_zero);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func) == 0);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func_zero) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func_zero) == 0);
}

TEST(nop_elimination, reshape_elimination_v1_dynamic)
{
    auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);
    auto abs = std::make_shared<op::Abs>(reshape_v1);
    auto f = std::make_shared<Function>(NodeVector{abs}, ParameterVector{arg, pattern});
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 1);
}

TEST(nop_elimination, concat_elimination_single_node)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A}, a), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
}

TEST(nop_elimination, concat_elimination_single_input)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto B = make_shared<op::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
}

TEST(nop_elimination, concat_elimination_single_input_dynamic)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3});
    auto B = make_shared<op::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
}
