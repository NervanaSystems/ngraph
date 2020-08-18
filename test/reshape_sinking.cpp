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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(reshape_sinking, edge_splitting)
{
    // checks if Reshapes are pushed through op::v0::Abs, but stopped by Sum
    Shape shape_nhwc{16, 28, 28, 1};
    Shape shape_nchw{16, 1, 28, 28};
    auto a = make_shared<op::v0::Parameter>(element::i32, shape_nhwc);
    auto reshape = make_shared<op::v0::Reshape>(a, AxisVector{0, 3, 1, 2}, shape_nchw);
    auto absn = make_shared<op::v0::Abs>(reshape);
    auto absn2 = make_shared<op::v0::Abs>(absn);
    auto sum = make_shared<op::v0::Sum>(reshape, AxisSet{0, 1, 2, 3});
    auto func = make_shared<Function>(OutputVector{absn2, sum}, ParameterVector{a});
    pass::Manager pass_manager;
    // size_t before_count = count_ops_of_type<op::v0::Reshape>(func);
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(func);
    ASSERT_EQ(func->get_results().at(1)->get_argument(0), sum);
    auto new_reshape = as_type_ptr<op::v0::Reshape>(func->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_reshape);
    ASSERT_EQ(new_reshape->get_output_shape(0), shape_nchw);
}

TEST(reshape_sinking, broadcast_swimming)
{
    Shape shape_nchw{1, 32, 536, 536};
    Shape shape_nhwc{1, 536, 536, 32};
    Shape shape_weights{16, 32, 3, 3};
    Shape conv_nhwc{1, 534, 534, 16};
    Shape conv_nchw{1, 16, 534, 534};
    AxisVector to_nhwc{0, 2, 3, 1};
    AxisVector to_nchw{0, 3, 1, 2};

    size_t channel = 16;
    auto bias = make_shared<op::v0::Parameter>(element::i32, Shape{channel});
    auto bias_reshape = make_shared<op::v0::Reshape>(bias, AxisVector{0}, Shape{1, channel});
    auto bias_broadcast = make_shared<op::v0::Broadcast>(bias_reshape, conv_nhwc, AxisSet{1, 2});

    auto input = make_shared<op::v0::Parameter>(element::i32, shape_nhwc);
    auto reshape_input = make_shared<op::v0::Reshape>(input, to_nchw, shape_nchw);

    auto weights = make_shared<op::v0::Parameter>(element::i32, shape_weights);
    auto conv = make_shared<op::v0::Convolution>(reshape_input, weights);
    auto conv_reshape = make_shared<op::v0::Reshape>(conv, to_nhwc, conv_nhwc);
    auto add = bias_broadcast + conv_reshape;
    auto relu = make_shared<op::v0::Relu>(add);

    auto func = make_shared<Function>(OutputVector{relu}, ParameterVector{bias, input, weights});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(func);

    ASSERT_EQ(add->get_output_shape(0), conv_nchw);
    ASSERT_EQ(add->get_input_shape(0), conv_nchw);
    ASSERT_EQ(add->get_argument(1), conv);
}

#ifndef NGRAPH_JSON_DISABLE
TEST(reshape_sinking, mnist_conv)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf_conv_mnist_nhwc.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::v0::Reshape>(func);
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    // pass_manager.register_pass<pass::CoreFusion>();
    // pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    size_t before_after = count_ops_of_type<op::v0::Reshape>(func);
    ASSERT_LE(before_after, before_count);
}
#endif

TEST(reshape_sinking, nasnet_pooladd)
{
    Shape input_shape{1, 3, 3, 1};

    auto input_type = element::f32;
    auto output_type = element::f32;

    auto X = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto c_weights = op::v0::Constant::create(input_type, Shape{1, 1, 1, 1}, {3});
    auto reshape1 = make_shared<op::v0::Reshape>(X, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});
    auto avgpool = make_shared<op::v0::AvgPool>(
        reshape1, Shape{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0});
    auto reshape2 =
        make_shared<op::v0::Reshape>(avgpool, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1});
    auto maxpool = make_shared<op::v0::MaxPool>(
        reshape1, Shape{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0});
    auto reshape3 =
        make_shared<op::v0::Reshape>(maxpool, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1});
    auto const1 = op::v0::Constant::create(input_type, Shape{1, 3, 3, 1}, {3});
    auto add1 = make_shared<op::v1::Add>(reshape3, const1);
    auto add2 = make_shared<op::v1::Add>(add1, reshape2);
    auto func = make_shared<Function>(add2, ParameterVector{X});

    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::v0::Reshape>(func);
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(func);
    size_t before_after = count_ops_of_type<op::v0::Reshape>(func);
    ASSERT_LE(before_after, before_count);
}

TEST(reshape_sinking, slice_pad)
{
    Shape shape_a{100, 8, 8, 1};

    AxisVector to_nhwc{0, 2, 3, 1};
    AxisVector to_nchw{0, 3, 1, 2};

    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto pad_value =
        op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    CoordinateDiff padding_below{0, 0, 0, 0};
    CoordinateDiff padding_above{0, 1, 1, 0};

    auto reshape1 = make_shared<op::v0::Reshape>(A, to_nchw, Shape{100, 1, 8, 8});
    auto maxpool = make_shared<op::v0::MaxPool>(
        reshape1, Shape{1, 1}, Strides{2, 2}, Shape{0, 0}, Shape{0, 0});
    auto reshape2 = make_shared<op::v0::Reshape>(maxpool, to_nhwc, Shape{100, 4, 4, 1});
    auto pad = make_shared<op::v0::Pad>(reshape2, pad_value, padding_below, padding_above);
    auto slice = make_shared<op::v0::Slice>(
        pad, Coordinate{0, 1, 1, 0}, Coordinate{100, 5, 5, 1}, Strides{1, 1, 1, 1});

    auto reshape3 = make_shared<op::v0::Reshape>(slice, to_nchw, Shape{100, 1, 4, 4});
    auto avgpool = make_shared<op::v0::AvgPool>(reshape3, Shape{1, 1}, Strides{2, 2});
    auto reshape4 = make_shared<op::v0::Reshape>(avgpool, to_nhwc, Shape{100, 1, 2, 2});
    auto f = make_shared<Function>(reshape4, ParameterVector{A});

    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::v0::Reshape>(f);
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    size_t before_after = count_ops_of_type<op::v0::Reshape>(f);
    ASSERT_LE(before_after, before_count);
}

TEST(reshape_sinking, concat)
{
    Shape shape{};
    Shape shape_w{1, 1, 1, 1};
    Shape shape_x{1, 3, 3, 1};
    Shape shape_b{1, 3, 3, 1};
    Shape r_shape{1, 3, 3, 2};

    auto B_ = op::v0::Constant::create(element::f32, shape_w, {3});
    auto B = make_shared<op::v0::Reshape>(B_, AxisVector{3, 2, 0, 1}, Shape{1, 1, 1, 1}); /* nchw */
    auto A_ = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto A = make_shared<op::v0::Reshape>(A_, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3}); /* nchw */
    auto C = op::v0::Constant::create(element::f32, Shape{1}, {2});
    auto R = make_shared<op::v0::Parameter>(element::f32, r_shape);

    auto conv = make_shared<op::v0::Convolution>(A,
                                                 B,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});
    auto reshape_conv =
        make_shared<op::v0::Reshape>(conv, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1}); /* nhwc */
    auto broadcast =
        make_shared<op::v0::Broadcast>(C, reshape_conv->get_output_shape(0), AxisSet{0, 1, 2});
    auto add = broadcast + reshape_conv;

    auto B1_ = op::v0::Constant::create(element::f32, shape_w, {3});
    auto B1 = make_shared<op::v0::Reshape>(B1_, AxisVector{3, 2, 0, 1}, Shape{1, 1, 1, 1});
    auto A1_ = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto A1 = make_shared<op::v0::Reshape>(A1_, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});
    auto C1 = op::v0::Constant::create(element::f32, Shape{1}, {2});
    auto R1 = make_shared<op::v0::Parameter>(element::f32, r_shape);

    auto conv1 = make_shared<op::v0::Convolution>(A1,
                                                  B1,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto reshape_conv1 =
        make_shared<op::v0::Reshape>(conv1, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1});
    auto broadcast1 =
        make_shared<op::v0::Broadcast>(C1, reshape_conv->get_output_shape(0), AxisSet{0, 1, 2});
    auto add1 = broadcast1 + reshape_conv1;

    auto concat = make_shared<op::v0::Concat>(OutputVector{add, add1}, 3);
    auto relu = make_shared<op::v0::Relu>(concat);
    auto reshape_relu =
        make_shared<op::v0::Reshape>(relu, AxisVector{0, 3, 1, 2}, Shape{1, 2, 3, 3}); /* nchw */
    auto B2_ = op::v0::Constant::create(element::f32, Shape{1, 1, 2, 1}, {2});
    auto B2 = make_shared<op::v0::Reshape>(B2_, AxisVector{3, 2, 0, 1}, Shape{1, 2, 1, 1});
    auto conv2 = make_shared<op::v0::Convolution>(reshape_relu,
                                                  B2,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto reshape_conv2 =
        make_shared<op::v0::Reshape>(conv2, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1}); /* nhwc */
    auto f = make_shared<Function>(reshape_conv2, ParameterVector{A_, A1_});
    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::v0::Reshape>(f);
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    size_t before_after = count_ops_of_type<op::v0::Reshape>(f);
    ASSERT_LE(before_after, before_count);
}

TEST(reshape_sinking, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::ReshapeSinking>();
    ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
