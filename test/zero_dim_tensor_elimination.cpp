/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto sum = std::make_shared<op::Sum>(A, AxisSet{0});
    auto abs_node = std::make_shared<op::Abs>(A);
    auto sum_node = std::make_shared<op::Sum>(abs_node, AxisSet{0});
    auto constant = std::make_shared<op::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(NodeVector{sum_node, constant}, op::ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(f);
    ASSERT_EQ(count_ops_of_type<op::Sum>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_conv)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 1, 0});
    auto weights = std::make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto convolution = std::make_shared<op::Convolution>(
        A, weights, Strides{1}, Strides{1}, CoordinateDiff{2}, CoordinateDiff{2});
    auto abs_node = std::make_shared<op::Abs>(convolution);
    auto constant = std::make_shared<op::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f =
        std::make_shared<Function>(NodeVector{abs_node, constant}, op::ParameterVector{A, weights});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(f);
    ASSERT_EQ(count_ops_of_type<op::Convolution>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_avg_pool)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 1, 0});

    auto avg_pool =
        std::make_shared<op::AvgPool>(A, Shape{1}, Strides{1}, Shape{2}, Shape{2}, true);
    auto abs_node = std::make_shared<op::Abs>(avg_pool);
    auto constant = std::make_shared<op::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(NodeVector{abs_node, constant}, op::ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(f);
    ASSERT_EQ(count_ops_of_type<op::AvgPool>(f), 0);
}

TEST(zero_dim_tensor_elimination, zero_const_pad)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::f32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{});

    auto pad = std::make_shared<op::Pad>(A, B, Shape{2}, Shape{2}, Shape{0});
    auto abs_node = std::make_shared<op::Abs>(pad);
    auto constant = std::make_shared<op::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(NodeVector{abs_node, constant}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(f);
    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 1);
}

TEST(zero_dim_tensor_elimination, zero_const_slice)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::f32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{});
    auto slice = make_shared<op::Slice>(A, Coordinate{0}, Coordinate{0});
    auto pad = std::make_shared<op::Pad>(A, B, Shape{2}, Shape{2}, Shape{0});
    auto abs_node = std::make_shared<op::Abs>(pad);
    auto constant = std::make_shared<op::Constant>(element::i32, zero_shape, std::vector<string>{});
    auto f = std::make_shared<Function>(NodeVector{abs_node, constant}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(f);
    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 0);
}
