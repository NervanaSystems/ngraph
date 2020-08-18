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
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(zero_dim_tensor_elimination, zero_sum)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::i32, zero_shape);
    auto abs_node = std::make_shared<op::v0::Abs>(A);
    auto sum_node = std::make_shared<op::v0::Sum>(abs_node, AxisSet{0});
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{sum_node, constant}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Sum>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Sum>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_product)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::i32, zero_shape);
    auto abs_node = std::make_shared<op::v0::Abs>(A);
    auto product_node = std::make_shared<op::v0::Product>(abs_node, AxisSet{0});
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{product_node, constant}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Product>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Product>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_min)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::i32, zero_shape);
    auto abs_node = std::make_shared<op::v0::Abs>(A);
    auto min_node = std::make_shared<op::v0::Min>(abs_node, AxisSet{0});
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{min_node, constant}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Min>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Min>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_max)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::i32, zero_shape);
    auto abs_node = std::make_shared<op::v0::Abs>(A);
    auto max_node = std::make_shared<op::v0::Max>(abs_node, AxisSet{0});
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{max_node, constant}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Max>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Max>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_conv)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 0});
    auto weights = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto convolution = std::make_shared<op::v0::Convolution>(
        A, weights, Strides{1}, Strides{1}, CoordinateDiff{2}, CoordinateDiff{2});
    auto abs_node = std::make_shared<op::v0::Abs>(convolution);
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f =
        std::make_shared<Function>(OutputVector{abs_node, constant}, ParameterVector{A, weights});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Convolution>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Convolution>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_avg_pool)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 0});

    auto avg_pool =
        std::make_shared<op::v0::AvgPool>(A, Shape{1}, Strides{1}, Shape{2}, Shape{2}, true);
    auto abs_node = std::make_shared<op::v0::Abs>(avg_pool);
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{abs_node, constant}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::AvgPool>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::AvgPool>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_pad)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::f32, zero_shape);
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{});

    auto pad = std::make_shared<op::v0::Pad>(A, B, CoordinateDiff{2}, CoordinateDiff{2});
    auto abs_node = std::make_shared<op::v0::Abs>(pad);
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{abs_node, constant}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 0);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 1);
}

TEST(zero_dim_tensor_elimination, zero_const_slice)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::v0::Parameter>(element::f32, zero_shape);
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto slice = make_shared<op::v0::Slice>(A, Coordinate{0}, Coordinate{0});
    auto pad = std::make_shared<op::v0::Pad>(A, B, CoordinateDiff{2}, CoordinateDiff{2});
    auto abs_node = std::make_shared<op::v0::Abs>(pad);
    auto constant =
        std::make_shared<op::v0::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(OutputVector{abs_node, constant}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 0);
    EXPECT_EQ(count_ops_of_type<op::v0::Slice>(f), 0);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Slice>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_argmax)
{
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{0, 2, 3});
    auto argmax = make_shared<op::v0::ArgMax>(A, 1, element::i32);
    auto f = std::make_shared<Function>(OutputVector{argmax}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::ArgMax>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::ArgMax>(f), 0);
    EXPECT_EQ(f->get_results().at(0)->get_output_shape(0), (Shape{0, 3}));
}

TEST(zero_dim_tensor_elimination, zero_argmin)
{
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{0, 2, 3});
    auto argmin = make_shared<op::v0::ArgMin>(A, 1, element::i32);
    auto f = std::make_shared<Function>(OutputVector{argmin}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    EXPECT_EQ(count_ops_of_type<op::v0::ArgMin>(f), 1);
    pass_manager.run_passes(f);
    EXPECT_EQ(count_ops_of_type<op::v0::ArgMin>(f), 0);
    EXPECT_EQ(f->get_results().at(0)->get_output_shape(0), (Shape{0, 3}));
}

TEST(zero_dim_tensor_elimination, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::ZeroDimTensorElimination>();
    ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
