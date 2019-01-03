//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(algebraic_simplification, add_types_shapes)
{
    Shape shapes[] = {Shape{}, Shape{2, 2}, Shape{3, 3, 3}};
    element::Type types[] = {element::i32, element::f32, element::f64};
    for (auto type : types)
    {
        for (auto shape : shapes)
        {
            pass::Manager pass_manager;
            pass_manager.register_pass<pass::AlgebraicSimplification>();

            auto a = make_shared<op::Parameter>(type, shape);
            auto b = make_shared<op::Parameter>(type, shape);
            auto c = make_shared<op::Parameter>(type, shape);
            auto iconst0 = ngraph::make_constant_from_string("0", type, shape);
            auto add_a_0 = a + iconst0;
            auto add_a_0_0 = add_a_0 + iconst0;
            auto add_b_0 = b + iconst0;
            auto add_b_0_0 = add_b_0 + iconst0;

            auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                                ParameterVector{a, b, c});
            pass_manager.run_passes(f);

            ASSERT_EQ(count_ops_of_type<op::Add>(f), 0);
            auto expected = ngraph::NodeVector{a, b, a, c, b};
            auto results = f->get_results();
            for (size_t i = 0; i < results.size(); i++)
            {
                ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
            }
        }
    }
}

TEST(algebraic_simplification, add_broadcast)
{
    Shape shape{2, 2};
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto c = make_shared<op::Parameter>(element::i32, shape);
    auto iconst0 = ngraph::make_zero(element::i32, Shape{});
    auto const_broadcast = make_shared<op::Broadcast>(iconst0, shape, AxisSet{0, 1});
    auto add_a_0 = a + const_broadcast;
    auto add_a_0_0 = add_a_0 + const_broadcast;
    auto add_b_0 = b + const_broadcast;
    auto add_b_0_0 = add_b_0 + const_broadcast;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Add>(f), 0);
    auto expected = ngraph::NodeVector{a, b, a, c, b};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++)
    {
        ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
    }
}

TEST(algebraic_simplification, multiply_broadcast)
{
    Shape shape{2, 2};
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto c = make_shared<op::Parameter>(element::i32, shape);
    auto iconst0 = ngraph::make_zero(element::i32, Shape{});
    auto const_broadcast = make_shared<op::Broadcast>(iconst0, shape, AxisSet{0, 1});
    auto mul_a_0 = a * const_broadcast;
    auto mul_a_0_0 = mul_a_0 * const_broadcast;
    auto mul_b_0 = b * const_broadcast;
    auto mul_b_0_0 = mul_b_0 * const_broadcast;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, mul_a_0_0, c, mul_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Add>(f), 0);
    auto expected = ngraph::NodeVector{a, b, const_broadcast, c, const_broadcast};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++)
    {
        ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
    }
}

TEST(algebraic_simplification, zero_plus_zero_commutativity)
{
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto iconst0 = ngraph::make_constant_from_string("0", type, shape);
    auto add_a_0 = iconst0 + iconst0;
    auto add_a_0_0 = iconst0 + iconst0;
    auto add_b_0 = iconst0 + b;
    auto add_b_0_0 = iconst0 + b;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    ASSERT_TRUE(ngraph::is_zero(f->get_results().at(2)->get_argument(0)));
    ASSERT_EQ(f->get_results().at(4)->get_argument(0), b);
}

TEST(algebraic_simplification, zero_multiply_zero_one)
{
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto iconst0 = ngraph::make_constant_from_string("0", type, shape);
    auto iconst1 = ngraph::make_constant_from_string("1", type, shape);
    auto add_a_0 = iconst0 * iconst0;
    auto add_b_0 = iconst1 * iconst0;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0, c, add_b_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    ASSERT_TRUE(ngraph::is_zero(f->get_results().at(2)->get_argument(0)));
    ASSERT_TRUE(ngraph::is_zero(f->get_results().at(4)->get_argument(0)));
}

TEST(algebraic_simplification, add_negative_tests)
{
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto abs_a = make_shared<op::Abs>(a);
    auto iconst2 = ngraph::make_constant_from_string("2", type, shape);
    auto add_a_0 = a + iconst2;
    auto add_a_0_0 = add_a_0 + iconst2;
    auto add_b_0 = b + abs_a;
    auto add_b_0_0 = add_b_0 + abs_a;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    auto expected = ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++)
    {
        ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
    }
}

TEST(algebraic_simplification, multiply_negative_tests)
{
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto abs_a = make_shared<op::Abs>(a);
    auto iconst2 = ngraph::make_constant_from_string("2", type, shape);
    auto add_a_0 = a * iconst2;
    auto add_a_0_0 = add_a_0 * iconst2;
    auto add_b_0 = b * abs_a;
    auto add_b_0_0 = add_b_0 * abs_a;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    auto expected = ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++)
    {
        ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
    }
}

TEST(algebraic_simplification, multiply_prod_vector_one)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{}, {2.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{3, 5}, AxisSet{0, 1});
    auto prod_fconst1 = std::make_shared<op::Product>(broadcast, AxisSet{1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{prod_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto new_broadcast =
        std::dynamic_pointer_cast<op::Broadcast>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_broadcast);
    auto new_const = std::dynamic_pointer_cast<op::Constant>(new_broadcast->get_argument(0));
    auto values = new_const->get_vector<double>();
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(values.at(0), 32);
}

TEST(algebraic_simplification, multiply_prod_scalar_one)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{}, {2.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{3, 5}, AxisSet{0, 1});
    auto prod_fconst1 = std::make_shared<op::Product>(broadcast, AxisSet{0, 1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{prod_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values = new_const->get_vector<double>();
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(values.at(0), 32768);
}

TEST(algebraic_simplification, multiply_prod_negative)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{2}, {1.0, 1.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{2, 5}, AxisSet{1});
    auto prod_fconst1 = std::make_shared<op::Product>(broadcast, AxisSet{0, 1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{prod_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto f_prod = f->get_results().at(0)->get_argument(0);
    ASSERT_EQ(f_prod, prod_fconst1);
}

TEST(algebraic_simplification, multiply_sum_scalar_one)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{}, {1.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{3, 5}, AxisSet{0, 1});
    auto sum_fconst1 = std::make_shared<op::Sum>(broadcast, AxisSet{0, 1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{sum_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values = new_const->get_vector<double>();
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(values.at(0), 15);
}

TEST(algebraic_simplification, multiply_sum_vector_one)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{}, {1.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{3, 5}, AxisSet{0, 1});
    auto sum_fconst1 = std::make_shared<op::Sum>(broadcast, AxisSet{1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{sum_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto new_broadcast =
        std::dynamic_pointer_cast<op::Broadcast>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_broadcast);
    auto new_const = std::dynamic_pointer_cast<op::Constant>(new_broadcast->get_argument(0));
    auto values = new_const->get_vector<double>();
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(values.at(0), 5);
}

TEST(algebraic_simplification, multiply_sum_negative)
{
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{2}, {1.0, 1.0});
    auto broadcast = std::make_shared<op::Broadcast>(fconst1, Shape{2, 5}, AxisSet{1});
    auto sum_fconst1 = std::make_shared<op::Sum>(broadcast, AxisSet{0, 1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{sum_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto f_sum = f->get_results().at(0)->get_argument(0);
    ASSERT_EQ(f_sum, sum_fconst1);
}

TEST(algebraic_simplification, concat_reshape_slice)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto goe = make_shared<op::GetOutputElement>(a, 0);
    auto slice1 = make_shared<op::Slice>(goe, Coordinate{0, 0}, Coordinate{32, 100}, Strides{1, 1});
    auto slice2 =
        make_shared<op::Slice>(goe, Coordinate{32, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 =
        make_shared<op::Slice>(goe, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    auto reshape1 = make_shared<op::Reshape>(slice1, AxisVector{0, 1}, Shape{32, 1, 100});
    auto reshape2 = make_shared<op::Reshape>(slice2, AxisVector{0, 1}, Shape{32, 1, 100});
    auto reshape3 = make_shared<op::Reshape>(slice3, AxisVector{0, 1}, Shape{32, 1, 100});

    size_t concat_axis = 1;
    auto concat = make_shared<op::Concat>(NodeVector{reshape1, reshape2, reshape3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Reshape>(f->get_results().at(0)->get_argument(0)));
}

TEST(algebraic_simplification, concat_slice)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto goe = make_shared<op::GetOutputElement>(a, 0);
    auto slice1 = make_shared<op::Slice>(goe, Coordinate{0, 0}, Coordinate{32, 100}, Strides{1, 1});
    auto slice2 =
        make_shared<op::Slice>(goe, Coordinate{32, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 =
        make_shared<op::Slice>(goe, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), goe);
}

TEST(algebraic_simplification, concat_parameter_slice)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto slice1 = make_shared<op::Slice>(a, Coordinate{0, 0}, Coordinate{32, 100}, Strides{1, 1});
    auto slice2 = make_shared<op::Slice>(a, Coordinate{32, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 = make_shared<op::Slice>(a, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), a);
}

TEST(algebraic_simplification, concat_parameter_slices_reversed)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto slice1 = make_shared<op::Slice>(a, Coordinate{0, 0}, Coordinate{32, 100}, Strides{1, 1});
    auto slice2 = make_shared<op::Slice>(a, Coordinate{32, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 = make_shared<op::Slice>(a, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice3, slice2, slice1}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), concat);
}

TEST(algebraic_simplification, concat_parameter_slices_element_count)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    //slicing 30 elements out of 96; should trigger a check that some elements are missing
    auto slice1 = make_shared<op::Slice>(a, Coordinate{0, 0}, Coordinate{10, 100}, Strides{1, 1});
    auto slice2 = make_shared<op::Slice>(a, Coordinate{10, 0}, Coordinate{20, 100}, Strides{1, 1});
    auto slice3 = make_shared<op::Slice>(a, Coordinate{20, 0}, Coordinate{30, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), concat);
}

TEST(algebraic_simplification, concat_parameter_non_uniform_slices)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto slice1 = make_shared<op::Slice>(a, Coordinate{0, 0}, Coordinate{38, 100}, Strides{1, 1});
    auto slice2 = make_shared<op::Slice>(a, Coordinate{38, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 = make_shared<op::Slice>(a, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), concat);
}

TEST(algebraic_simplification, concat_different_goes)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto goe1 = make_shared<op::GetOutputElement>(a, 0);
    auto goe2 = make_shared<op::GetOutputElement>(a, 0);
    auto slice1 =
        make_shared<op::Slice>(goe1, Coordinate{0, 0}, Coordinate{32, 100}, Strides{1, 1});
    auto slice2 =
        make_shared<op::Slice>(goe2, Coordinate{32, 0}, Coordinate{64, 100}, Strides{1, 1});
    auto slice3 =
        make_shared<op::Slice>(goe1, Coordinate{64, 0}, Coordinate{96, 100}, Strides{1, 1});

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), concat);
}

TEST(algebraic_simplification, log_neg_neg)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto b = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto exp_a = make_shared<op::Exp>(a);
    auto div = exp_a / b;
    auto log_div = make_shared<op::Log>(div);

    auto neg_inner = make_shared<op::Negative>(log_div);
    auto neg2 = make_shared<op::Negative>(neg_inner);
    auto neg3 = make_shared<op::Negative>(neg2);
    auto neg4 = make_shared<op::Negative>(neg3);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{neg4}, ParameterVector{a, b});
    pass_manager.run_passes(f);
    auto sub = std::dynamic_pointer_cast<op::Subtract>(neg_inner->get_argument(0));
    ASSERT_TRUE(sub != nullptr);
    ASSERT_EQ(sub->get_argument(0), a);
    auto new_log = std::dynamic_pointer_cast<op::Log>(sub->get_argument(1));
    ASSERT_TRUE(new_log != nullptr);
    ASSERT_EQ(new_log->get_argument(0), b);
}

TEST(algebraic_simplification, log_no_exp)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto b = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto abs_a = make_shared<op::Abs>(a);
    auto div = abs_a / b;
    auto log_div = make_shared<op::Log>(div);

    auto neg_inner = make_shared<op::Negative>(log_div);
    auto neg2 = make_shared<op::Negative>(neg_inner);
    auto neg3 = make_shared<op::Negative>(neg2);
    auto neg4 = make_shared<op::Negative>(neg3);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{neg4}, ParameterVector{a, b});
    pass_manager.run_passes(f);
    ASSERT_EQ(neg_inner->get_argument(0), log_div);
}

TEST(algebraic_simplification, log_no_divide)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto b = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto exp_a = make_shared<op::Exp>(a);
    auto mul = exp_a * b;
    auto log_mul = make_shared<op::Log>(mul);

    auto neg_inner = make_shared<op::Negative>(log_mul);
    auto neg2 = make_shared<op::Negative>(neg_inner);
    auto neg3 = make_shared<op::Negative>(neg2);
    auto neg4 = make_shared<op::Negative>(neg3);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{neg4}, ParameterVector{a, b});
    pass_manager.run_passes(f);
    ASSERT_EQ(neg_inner->get_argument(0), log_mul);
}
