//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
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
    //checks if Reshapes are pushed through op::Abs, but stopped by Sum
    Shape shape_nhwc{16, 28, 28, 1};
    Shape shape_nchw{16, 1, 28, 28};
    auto a = make_shared<op::Parameter>(element::i32, shape_nhwc);
    auto reshape = make_shared<op::Reshape>(a, AxisVector{0, 3, 1, 2}, shape_nchw);
    auto absn = make_shared<op::Abs>(reshape);
    auto absn2 = make_shared<op::Abs>(absn);
    auto sum = make_shared<op::Sum>(reshape, AxisSet{0, 1, 2, 3});
    auto func = make_shared<Function>(NodeVector{absn2, sum}, ParameterVector{a});
    pass::Manager pass_manager;
    //size_t before_count = count_ops_of_type<op::Reshape>(func);
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);
    ASSERT_EQ(func->get_results().at(1)->get_argument(0), sum);
    auto new_reshape =
        std::dynamic_pointer_cast<op::Reshape>(func->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_reshape);
    ASSERT_EQ(new_reshape->get_shape(), shape_nchw);
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
    auto bias = make_shared<op::Parameter>(element::i32, Shape{channel});
    auto bias_reshape = make_shared<op::Reshape>(bias, AxisVector{0}, Shape{1, channel});
    auto bias_broadcast = make_shared<op::Broadcast>(bias_reshape, conv_nhwc, AxisSet{1, 2});

    auto input = make_shared<op::Parameter>(element::i32, shape_nhwc);
    auto reshape_input = make_shared<op::Reshape>(input, to_nchw, shape_nchw);

    auto weights = make_shared<op::Parameter>(element::i32, shape_weights);
    auto conv = make_shared<op::Convolution>(reshape_input, weights);
    auto conv_reshape = make_shared<op::Reshape>(conv, to_nhwc, conv_nhwc);
    auto add = bias_broadcast + conv_reshape;
    auto relu = make_shared<op::Relu>(add);

    auto func = make_shared<Function>(NodeVector{relu}, ParameterVector{bias, input, weights});
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(func);

    ASSERT_EQ(add->get_shape(), conv_nchw);
    ASSERT_EQ(add->get_argument(0)->get_shape(), conv_nchw);
    ASSERT_EQ(add->get_argument(1), conv);
}

TEST(reshape_sinking, mnist_conv)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf_conv_mnist_nhwc.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::Reshape>(func);
    //pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    //pass_manager.register_pass<pass::CoreFusion>();
    //pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    //pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);
    size_t before_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_LE(before_after, before_count);
}

TEST(reshape_sinking, nasnet_pooladd)
{
    Shape input_shape{1, 3, 3, 1};

    auto input_type = element::f32;
    auto output_type = element::f32;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto c_weights = op::Constant::create(input_type, Shape{1, 1, 1, 1}, {3});
    auto reshape1 = make_shared<op::Reshape>(X, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});
    auto avgpool =
        make_shared<op::AvgPool>(reshape1, Shape{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0});
    auto reshape2 = make_shared<op::Reshape>(avgpool, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1});
    auto maxpool =
        make_shared<op::MaxPool>(reshape1, Shape{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0});
    auto reshape3 = make_shared<op::Reshape>(maxpool, AxisVector{0, 2, 3, 1}, Shape{1, 3, 3, 1});
    auto const1 = op::Constant::create(input_type, Shape{1, 3, 3, 1}, {3});
    auto add1 = make_shared<op::Add>(reshape3, const1);
    auto add2 = make_shared<op::Add>(add1, reshape2);
    auto func = make_shared<Function>(add2, ParameterVector{X});

    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::Reshape>(func);
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::ReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);
    size_t before_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_LE(before_after, before_count);
}
