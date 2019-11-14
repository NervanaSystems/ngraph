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
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
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
    auto func = make_shared<Function>(graph, ParameterVector{B});
    pass_manager.run_passes(func);
    ASSERT_NE(as_type_ptr<op::Relu>(graph->get_argument(0)), nullptr);
}

#ifndef NGRAPH_JSON_DISABLE
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
#endif

TEST(core_fusion, sigmoid_fprop_fusion_no_broadcast)
{
    auto make_function = []() {
        auto input = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});
        auto neg_input = std::make_shared<op::Negative>(input);
        auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

        auto constant =
            op::Constant::create(element::f32, Shape{3, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        auto add_exp = std::make_shared<op::Add>(exp_neg_input, constant);
        auto divide_1_over_exp = std::make_shared<op::Divide>(constant, add_exp);
        return make_shared<Function>(NodeVector{divide_1_over_exp}, ParameterVector{input});
    };
    auto func = make_function();

    // Check fusion happens
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::Sigmoid>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(core_fusion, sigmoid_fprop_fusion_no_broadcast2)
{
    auto make_function = []() {
        auto input = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});
        auto neg_input = std::make_shared<op::Negative>(input);
        auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

        auto constant =
            op::Constant::create(element::f32, Shape{3, 4}, {1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1});

        auto add_exp = std::make_shared<op::Add>(exp_neg_input, constant);
        auto divide_1_over_exp = std::make_shared<op::Divide>(constant, add_exp);
        return make_shared<Function>(NodeVector{divide_1_over_exp}, ParameterVector{input});
    };
    auto func = make_function();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::Sigmoid>(func);
    ASSERT_EQ(ccg, 0);
}

TEST(core_fusion, reshape_broadcast)
{
    auto generate_func = []() {
        auto input = make_shared<op::Parameter>(element::f32, Shape{10});
        auto reshape1 = make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, 10, 1});
        auto broadcast =
            make_shared<op::Broadcast>(reshape1, Shape{1, 5, 10, 8, 1, 20}, AxisSet{1, 3, 5});
        auto f = make_shared<Function>(broadcast, ParameterVector{input});
        return f;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
}

TEST(core_fusion, reshape_broadcast_graph_optimized)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{10});
    auto reshape1 = make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, 10, 1});
    auto broadcast =
        make_shared<op::Broadcast>(reshape1, Shape{1, 5, 10, 8, 1, 20}, AxisSet{1, 3, 5});
    auto optimized_f = make_shared<Function>(broadcast, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(optimized_f);

    auto new_broadcast =
        as_type_ptr<op::Broadcast>(optimized_f->get_results().at(0)->get_argument(0));
    EXPECT_EQ(new_broadcast->get_argument(0), input);
    EXPECT_EQ(new_broadcast->get_broadcast_axes(), (AxisSet{0, 1, 3, 4, 5}));
}

TEST(core_fusion, reshape_broadcast_adds_one)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{10});
    auto reshape1 = make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, 10, 1});
    auto broadcast =
        make_shared<op::Broadcast>(reshape1, Shape{1, 5, 10, 8, 1, 20, 1}, AxisSet{1, 3, 5, 6});
    auto optimized_f = make_shared<Function>(broadcast, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(optimized_f);

    auto new_broadcast =
        as_type_ptr<op::Broadcast>(optimized_f->get_results().at(0)->get_argument(0));
    EXPECT_EQ(new_broadcast, broadcast);
    EXPECT_EQ(new_broadcast->get_argument(0), reshape1);
}

TEST(core_fusion, reshape_broadcast_wrong_reshape)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{10});
    auto reshape1 = make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, 5, 2});
    auto broadcast =
        make_shared<op::Broadcast>(reshape1, Shape{1, 5, 5, 8, 2, 20}, AxisSet{1, 3, 5});
    auto optimized_f = make_shared<Function>(broadcast, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(optimized_f);

    auto new_broadcast =
        as_type_ptr<op::Broadcast>(optimized_f->get_results().at(0)->get_argument(0));
    EXPECT_EQ(new_broadcast, broadcast);
    EXPECT_EQ(new_broadcast->get_argument(0), reshape1);
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
    auto params = ParameterVector{data_stride3,
                                  weights_stride3,
                                  param_broadcast_w3,
                                  weights_stride1,
                                  param_broadcast_w1,
                                  other_arg,
                                  weights_conv_s2};
    auto func = make_shared<Function>(NodeVector{conv_s2_1, conv_s2_2}, params);
    pass_manager.run_passes(func);
    auto results = func->get_results();
    auto t_eltwise_conv1 = as_type_ptr<op::Convolution>(results.at(0)->get_argument(0));
    auto t_eltwise_conv2 = as_type_ptr<op::Convolution>(results.at(1)->get_argument(0));
    ASSERT_TRUE(t_eltwise_conv1);
    ASSERT_TRUE(t_eltwise_conv2);
    ASSERT_EQ(t_eltwise_conv1->get_window_movement_strides(), stride_1);
    ASSERT_EQ(t_eltwise_conv2->get_window_movement_strides(), stride_1);
}

static std::shared_ptr<Function> generate_reshape_softmax_reshape()
{
    Shape shape_nchw{10, 20, 30, 40};
    Shape shape_nhwc{10, 30, 40, 20};
    AxisVector to_nhwc{0, 2, 3, 1};
    AxisVector to_nchw{0, 3, 1, 2};
    auto input = make_shared<op::Parameter>(element::f32, shape_nchw);
    auto reshape1 = make_shared<op::Reshape>(input, to_nhwc, shape_nhwc);
    auto softmax = make_shared<op::Softmax>(reshape1, AxisSet{1, 2, 3});
    auto reshape2 = make_shared<op::Reshape>(softmax, to_nchw, shape_nchw);
    auto f = make_shared<Function>(reshape2, ParameterVector{input});
    return f;
}

TEST(core_fusion, reshape_softmax_reshape)
{
    auto baseline_f = generate_reshape_softmax_reshape();
    auto optimized_f = generate_reshape_softmax_reshape();
    auto baseline_input = baseline_f->get_parameters().at(0);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input->get_shape()));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
}

TEST(core_fusion, zero_padded_reshaped_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 1});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, CoordinateDiff{0, 1, 0, 0}, CoordinateDiff{0, 0, 1, 0});

    auto reshape = make_shared<op::Reshape>(pad, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});

    auto conv = make_shared<op::Convolution>(reshape,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(core_fusion, zero_padded_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::Convolution>(pad,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(core_fusion, non_zero_padded_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{1.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::Convolution>(pad,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);
}

TEST(core_fusion, zero_padded_conv_backprop_filters)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::ConvolutionBackpropFilters>(pad,
                                                            Shape{1, 1, 2, 2},
                                                            F,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(core_fusion, conv_bias)
{
    auto gen_f = [](bool with_fused_op) {
        auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto weights = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
        auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
        if (with_fused_op)
        {
            return make_shared<Function>(make_shared<op::ConvolutionBias>(data, weights, bias),
                                         ParameterVector{data, weights, bias});
        }
        else
        {
            auto conv = make_shared<op::Convolution>(data, weights);
            auto conv_bias =
                conv + make_shared<op::Broadcast>(bias, conv->get_shape(), AxisSet{0, 2, 3});
            return make_shared<Function>(conv_bias, ParameterVector{data, weights, bias});
        }
    };

    auto fused_f = gen_f(true);
    auto decomp_f1 = gen_f(false);
    auto decomp_f2 = gen_f(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>(ngraph::pass::FusionType::ALL_FUSIONS);
    pass_manager.run_passes(decomp_f1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBias>(decomp_f1), 1);

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : fused_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto fused_r = execute(fused_f, args, "INTERPRETER");
    auto decomp_r1 = execute(decomp_f1, args, "INTERPRETER");
    auto decomp_r2 = execute(decomp_f2, args, "INTERPRETER");

    for (size_t i = 0; i < fused_r.size(); i++)
    {
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r1.at(i)));
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r2.at(i)));
    }
}

TEST(core_fusion, conv_bias_bcast_reshape)
{
    // PaddlePaddle pattern
    auto gen_f = [](bool with_fused_op) {
        auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto weights = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
        auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
        if (with_fused_op)
        {
            return make_shared<Function>(make_shared<op::ConvolutionBias>(data, weights, bias),
                                         ParameterVector{data, weights, bias});
        }
        else
        {
            auto conv = make_shared<op::Convolution>(data, weights);
            auto bias_bcast = make_shared<op::Broadcast>(bias, Shape{2, 4, 12}, AxisSet{0, 2});
            auto conv_bias =
                conv + make_shared<op::Reshape>(bias_bcast, AxisVector{0, 1, 2}, conv->get_shape());
            return make_shared<Function>(conv_bias, ParameterVector{data, weights, bias});
        }
    };

    auto fused_f = gen_f(true);
    auto decomp_f1 = gen_f(false);
    auto decomp_f2 = gen_f(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>(ngraph::pass::FusionType::ALL_FUSIONS);
    pass_manager.run_passes(decomp_f1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBias>(decomp_f1), 1);

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : fused_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto fused_r = execute(fused_f, args, "INTERPRETER");
    auto decomp_r1 = execute(decomp_f1, args, "INTERPRETER");
    auto decomp_r2 = execute(decomp_f2, args, "INTERPRETER");

    for (size_t i = 0; i < fused_r.size(); i++)
    {
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r1.at(i)));
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r2.at(i)));
    }
}

TEST(core_fusion, conv_bias_add)
{
    auto gen_f = [](bool with_fused_op) {
        auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto weights = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
        auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
        auto add = make_shared<op::Parameter>(element::f32, Shape{2, 4, 3, 4});
        if (with_fused_op)
        {
            auto conv_bias = make_shared<op::ConvolutionBias>(data, weights, bias);
            return make_shared<Function>(make_shared<op::ConvolutionBiasAdd>(conv_bias, add),
                                         ParameterVector{data, weights, bias, add});
        }
        else
        {
            auto conv = make_shared<op::Convolution>(data, weights);
            auto conv_bias =
                conv + make_shared<op::Broadcast>(bias, conv->get_shape(), AxisSet{0, 2, 3});
            return make_shared<Function>(conv_bias + add,
                                         ParameterVector{data, weights, bias, add});
        }
    };

    auto fused_f = gen_f(true);
    auto decomp_f1 = gen_f(false);
    auto decomp_f2 = gen_f(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>(ngraph::pass::FusionType::ALL_FUSIONS);
    pass_manager.run_passes(decomp_f1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBiasAdd>(decomp_f1), 1);

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : fused_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto fused_r = execute(fused_f, args, "INTERPRETER");
    auto decomp_r1 = execute(decomp_f1, args, "INTERPRETER");
    auto decomp_r2 = execute(decomp_f2, args, "INTERPRETER");

    for (size_t i = 0; i < fused_r.size(); i++)
    {
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r1.at(i)));
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r2.at(i)));
    }
}

// TODO: Enable once fusion is moved to core
TEST(core_fusion, DISABLED_conv_bias_bprop)
{
    auto gen_f = [](bool with_fused_op) {
        auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto weights = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
        auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
        auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 4, 3, 4});
        if (with_fused_op)
        {
            auto conv_bprop =
                make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                                    weights->get_shape(),
                                                                    bias->get_shape(),
                                                                    delta,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1});
            auto goe0 = make_shared<op::GetOutputElement>(conv_bprop, 0);
            auto goe1 = make_shared<op::GetOutputElement>(conv_bprop, 1);
            return make_shared<Function>(NodeVector{goe0, goe1}, ParameterVector{data, delta});
        }
        else
        {
            auto conv_bprop = make_shared<op::ConvolutionBackpropFilters>(data,
                                                                          weights->get_shape(),
                                                                          delta,
                                                                          Strides{1, 1},
                                                                          Strides{1, 1},
                                                                          CoordinateDiff{0, 0},
                                                                          CoordinateDiff{0, 0},
                                                                          Strides{1, 1});
            auto bias_bprop = make_shared<op::Sum>(delta, AxisSet{0, 2, 3});
            return make_shared<Function>(NodeVector{conv_bprop, bias_bprop},
                                         ParameterVector{data, delta});
        }
    };

    auto fused_f = gen_f(true);
    auto decomp_f1 = gen_f(false);
    auto decomp_f2 = gen_f(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>(ngraph::pass::FusionType::ALL_FUSIONS);
    pass_manager.run_passes(decomp_f1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBiasBackpropFiltersBias>(decomp_f1), 1);

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : fused_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto fused_r = execute(fused_f, args, "INTERPRETER");
    auto decomp_r1 = execute(decomp_f1, args, "INTERPRETER");
    auto decomp_r2 = execute(decomp_f2, args, "INTERPRETER");

    for (size_t i = 0; i < fused_r.size(); i++)
    {
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r1.at(i)));
        EXPECT_TRUE(test::all_close(fused_r.at(i), decomp_r2.at(i)));
    }
}

TEST(batch_fusion, group_convolution_fusion)
{
    Shape shape_a{1, 32, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};

    auto a_slice0 = std::make_shared<op::Slice>(A, Coordinate{0, 0, 0, 0}, Coordinate{1, 16, 2, 2});
    auto a_slice1 =
        std::make_shared<op::Slice>(A, Coordinate{0, 16, 0, 0}, Coordinate{1, 32, 2, 2});

    auto b_slice0 = std::make_shared<op::Slice>(B, Coordinate{0, 0, 0, 0}, Coordinate{1, 16, 1, 1});
    auto b_slice1 = std::make_shared<op::Slice>(B, Coordinate{1, 0, 0, 0}, Coordinate{2, 16, 1, 1});

    auto conv_lower = make_shared<op::Convolution>(a_slice0,
                                                   b_slice0,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});

    auto conv_upper = make_shared<op::Convolution>(a_slice1,
                                                   b_slice1,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});

    auto concat = make_shared<op::Concat>(NodeVector{conv_lower, conv_upper}, 1);

    auto f = make_shared<Function>(NodeVector{concat}, ParameterVector{A, B});
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::BatchFusion>();
    pass_manager.run_passes(f);
    auto gc = as_type_ptr<op::GroupConvolution>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(gc);
}

TEST(core_fusion, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::CoreFusion>();
    ASSERT_FALSE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(batch_fusion, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::BatchFusion>();
    ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

#ifndef NGRAPH_JSON_DISABLE
TEST(core_fusion, softmax_crossentropy_fprop_1)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-function3.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    // during this optimization for numeric stability we will reduce softmax operation to
    // - summation (labels (input - max(input) - log (summation(exp ^ (input - max(input)))
    // count_of(softmax) should be equal to zero if fusion is successful
    size_t softmax = count_ops_of_type<op::Softmax>(cpu_f);
    ASSERT_EQ(softmax, 0);
}

TEST(core_fusion, softmax_crossentropy_fprop_2)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-function1.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    // during this optimization for numeric stability we will reduce softmax operation to
    // - summation (labels (input - max(input) - log (summation(exp ^ (input - max(input)))
    // count_of(softmax) should be equal to zero if fusion is successful
    size_t softmax = count_ops_of_type<op::Softmax>(cpu_f);
    ASSERT_EQ(softmax, 0);
}

TEST(core_fusion, softmax_crossentropy_bprop_with_soft_labels)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-bprop0.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }

    // during this optimization for numeric stability we will eliminate (softmax / softmax)
    // the number of div operator for cpu_f should be zero if the fusion is valid
    size_t divide = count_ops_of_type<op::Divide>(cpu_f);
    ASSERT_EQ(divide, 0);
}

TEST(core_fusion, softmax_crossentropy_bprop_with_ignore_mask)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-bprop1.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }

    // during this optimization for numeric stability we will eliminate (softmax / softmax)
    // the number of div operator for cpu_f should be zero if the fusion is valid
    size_t divide = count_ops_of_type<op::Divide>(cpu_f);
    ASSERT_EQ(divide, 0);
}
#endif

void test_softmax_crossentropy(Shape input_shape,
                               Shape label_shape,
                               bool soft_label,
                               int64_t ignore_index)
{
    auto input = std::make_shared<op::Parameter>(element::f64, input_shape);
    auto labels = std::make_shared<op::Parameter>(element::i64, label_shape);
    auto sm_ce = std::make_shared<op::SoftmaxCrossEntropy>(input, labels, soft_label, ignore_index);
    auto cpu_f = make_shared<Function>(sm_ce, ParameterVector{input, labels});

    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto cpu_results = execute(cpu_f, args, "CPU");
    // if softlabels = flase, we will have one one hot encoding for labels
    if (!soft_label)
    {
        size_t onehot = count_ops_of_type<op::OneHot>(cpu_f);
        ASSERT_EQ(onehot, 1);
    }
    if (ignore_index >= 0 && !soft_label)
    // check for the mask
    {
        size_t not_equal = count_ops_of_type<op::NotEqual>(cpu_f);
        ASSERT_EQ(not_equal, 1);
    }
}

TEST(core_fusion, softmax_crossentropy)
{
    test_softmax_crossentropy(Shape{41, 37}, Shape{41, 37}, true, -1);
    test_softmax_crossentropy(Shape{41, 37}, Shape{41, 1}, false, 5);
}
