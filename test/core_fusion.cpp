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
#include "ngraph/op/relu.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(core_fusion, core_fusion_pass_basic)
{
    auto shape_a = Shape{1, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto graph = make_shared<op::Abs>(max);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{B});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::Relu>(graph->get_argument(0)), nullptr);
}

TEST(core_fusion, sigmoid_fprop_fusion)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::Sigmoid>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(core_fusion, sigmoid_bprop_fusion)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    auto df = autodiff::backprop_function(func);
    auto backend = runtime::Backend::create("CPU");
    backend->compile(df);
    size_t ccg = count_ops_of_type<op::SigmoidBackprop>(df);
    ASSERT_EQ(ccg, 1);
}

TEST(core_fusion, sparsity_opt_56x56)
{
    Shape win_size_3{1, 1, 3, 3};
    Shape win_size_1{1, 1, 1, 1};
    Strides stride_2{2, 2};
    Strides stride_1{1, 1};
    CoordinateDiff pad_0{0, 0};
    CoordinateDiff pad_1{1, 1};
    auto data_stride3 = std::make_shared<op::Parameter>(element::f32, Shape{1, 64, 56, 56});
    auto weights_stride3 = std::make_shared<op::Parameter>(element::f32, Shape{64, 64, 3, 3});

    auto conv_stride3 = std::make_shared<op::Convolution>(
        data_stride3, weights_stride3, stride_1, stride_1, pad_1, pad_1);
    auto param_broadcast_w3 = std::make_shared<op::Parameter>(element::f32, Shape{64});
    auto broadcast_w3 =
        std::make_shared<op::Broadcast>(param_broadcast_w3, Shape{1, 64, 56, 56}, AxisSet{0, 2, 3});
    auto add_w3 = std::make_shared<op::Add>(conv_stride3, broadcast_w3);
    auto relu_w3 = std::make_shared<op::Relu>(add_w3);
    ///
    auto weights_stride1 = std::make_shared<op::Parameter>(element::f32, Shape{256, 64, 1, 1});
    auto conv_stride1 = std::make_shared<op::Convolution>(relu_w3, weights_stride1);
    auto param_broadcast_w1 = std::make_shared<op::Parameter>(element::f32, Shape{256});
    auto broadcast_w1 = std::make_shared<op::Broadcast>(
        param_broadcast_w1, Shape{1, 256, 56, 56}, AxisSet{0, 2, 3});
    auto add_w1 = std::make_shared<op::Add>(conv_stride1, broadcast_w1);
    ////
    auto other_arg = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 56, 56});
    auto add_two_convs = std::make_shared<op::Add>(add_w1, other_arg);
    auto relu_two_convs = std::make_shared<op::Relu>(add_two_convs);
    ///
    auto weights_conv_s2 = std::make_shared<op::Parameter>(element::f32, Shape{512, 256, 1, 1});
    auto conv_s2_1 = std::make_shared<op::Convolution>(relu_two_convs, weights_conv_s2, stride_2);
    auto conv_s2_2 = std::make_shared<op::Convolution>(relu_two_convs, weights_conv_s2, stride_2);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    auto params = op::ParameterVector{data_stride3,
                                      weights_stride3,
                                      param_broadcast_w3,
                                      weights_stride1,
                                      param_broadcast_w1,
                                      other_arg,
                                      weights_conv_s2};
    auto func = make_shared<Function>(NodeVector{conv_s2_1, conv_s2_2}, params);
    pass_manager.run_passes(func);
    auto results = func->get_results();
    auto t_eltwise_conv1 =
        std::dynamic_pointer_cast<op::Convolution>(results.at(0)->get_argument(0));
    auto t_eltwise_conv2 =
        std::dynamic_pointer_cast<op::Convolution>(results.at(1)->get_argument(0));
    ASSERT_TRUE(t_eltwise_conv1);
    ASSERT_TRUE(t_eltwise_conv2);
    ASSERT_EQ(t_eltwise_conv1->get_window_movement_strides(), stride_1);
    ASSERT_EQ(t_eltwise_conv2->get_window_movement_strides(), stride_1);
}
