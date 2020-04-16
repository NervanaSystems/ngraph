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
#include "util/all_close.hpp"
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
    auto p = make_shared<op::v0::Pad>(A, B, padding_below, padding_above);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(p), ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(f), 0);
}

TEST(nop_elimination, eliminate_sum)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::Sum>(A, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Sum>(f), 0);
}

TEST(nop_elimination, eliminate_convert)
{
    Shape shape{};
    auto type = element::f32;
    auto A = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::v0::Convert>(A, element::f32);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(c), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(nop_elimination, convert_type_agnostic)
{
    Shape shape{};
    auto type = element::from<char>();
    auto A = make_shared<op::Parameter>(type, shape);
    auto c1 = make_shared<op::v0::Convert>(A, element::from<uint8_t>());
    auto c = make_shared<op::v0::Convert>(c1, element::f32);
    auto z = make_shared<op::v3::NonZero>(c);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(z), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(nop_elimination, eliminate_slice)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::Slice>(A, Coordinate{0, 0}, Coordinate{2, 2});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Slice>(f), 0);
}

TEST(nop_elimination, eliminate_broadcast)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::v0::Broadcast>(A, shape, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(b), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 0);
}

TEST(nop_elimination, eliminate_stop_gradient)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::StopGradient>(A);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::StopGradient>(f), 0);
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
        auto pattern_org = op::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto pattern = op::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto reshape_v1_org = std::make_shared<op::v1::Reshape>(arg, pattern_org, zero);
        auto reshape_v1 = std::make_shared<op::v1::Reshape>(reshape_v1_org, pattern, zero);
        auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
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
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func_zero) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func_zero) == 1);
}

TEST(nop_elimination, reshape_elimination_v1_dynamic)
{
    auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);
    auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
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
    auto f =
        make_shared<Function>(make_shared<op::v0::Concat>(NodeVector{A}, a), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 1);
}

TEST(nop_elimination, concat_elimination_single_input)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, concat_elimination_single_input_dynamic)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, squeeze_unsqueeze_elimination)
{
    auto generate_func = [](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Squeeze>(A1, axes);
        auto B1 = make_shared<op::v0::Unsqueeze>(B, axes);
        return make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
    };

    auto check_usecase = [&](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto baseline_f = generate_func(shape, axes_val);
        auto optimized_f = generate_func(shape, axes_val);
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.run_passes(optimized_f);

        ASSERT_EQ(baseline_f->get_results()[0]->get_shape(),
                  optimized_f->get_results()[0]->get_shape());

        vector<vector<float>> args{vector<float>{5.5, 6.4, 7.9, 3.4, 2.3, 1.1}};
        auto baseline_results = execute(baseline_f, args, "INTERPRETER");
        auto optimized_results = execute(optimized_f, args, "INTERPRETER");
        EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{1, 6}, std::vector<int64_t>{0});
    check_usecase(Shape{1, 3, 2, 1}, std::vector<int64_t>{0, 3});
    check_usecase(Shape{1, 3, 1, 2, 1}, std::vector<int64_t>{0, 2, 4});
    check_usecase(Shape{1, 3, 2, 1}, std::vector<int64_t>{-1, -4});
}

TEST(nop_elimination, unsqueeze_squeeze_elimination)
{
    auto generate_func = [](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Unsqueeze>(A1, axes);
        auto B1 = make_shared<op::v0::Squeeze>(B, axes);
        return make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
    };

    auto check_usecase = [&](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto baseline_f = generate_func(shape, axes_val);
        auto optimized_f = generate_func(shape, axes_val);
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.run_passes(optimized_f);

        ASSERT_EQ(baseline_f->get_results()[0]->get_shape(),
                  optimized_f->get_results()[0]->get_shape());

        vector<vector<float>> args{vector<float>{5.5, 6.4, 7.9, 3.4, 2.3, 1.1}};
        auto baseline_results = execute(baseline_f, args, "INTERPRETER");
        auto optimized_results = execute(optimized_f, args, "INTERPRETER");
        EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{6}, std::vector<int64_t>{0});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 3});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 2, 4});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{-1, -4});
}
