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
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_mat_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

#include "util/random.hpp"

using namespace ngraph;
using namespace std;

TEST(cpu_fusion, gemm_pattern)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::Dot>(A, B);
    auto broadcast = make_shared<op::Broadcast>(C, dot->get_shape(), AxisSet{0});
    auto add = dot + broadcast;

    auto W = std::make_shared<pattern::op::Label>(A);
    auto x = std::make_shared<pattern::op::Label>(B);

    auto reshape_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Reshape>(n));
    };

    auto skip_w = std::make_shared<pattern::op::Any>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Any>(x, reshape_pred);

    auto pdot = make_shared<op::Dot>(skip_w, skip_x);
    auto b = std::make_shared<pattern::op::Label>(C);
    auto pbroadcast = make_shared<op::Broadcast>(b, dot->get_shape(), AxisSet{0});
    auto padd = pdot + pbroadcast;

    TestMatcher n(nullptr);
    ASSERT_TRUE(n.match(padd, add));
    ASSERT_EQ(n.get_pattern_map()[W], A);
    ASSERT_EQ(n.get_pattern_map()[x], B);
    ASSERT_EQ(n.get_pattern_map()[b], C);

    auto reshape_w = make_shared<op::Reshape>(A, AxisVector{1, 0}, W->get_shape());
    auto reshape_x = make_shared<op::Reshape>(B, AxisVector{1, 0}, x->get_shape());
    auto re_dot = make_shared<op::Dot>(reshape_w, reshape_x);
    auto re_add = re_dot + broadcast;
    ASSERT_TRUE(n.match(padd, re_add));
    ASSERT_EQ(n.get_pattern_map()[W], A);
    ASSERT_EQ(n.get_pattern_map()[x], B);
    ASSERT_EQ(n.get_pattern_map()[b], C);

    auto cg = make_shared<op::MatmulBias>(
        W, x, C, W->get_shape(), x->get_shape(), false, false, AxisSet{0});
}

TEST(cpu_fusion, gemm_cpu_broadcast_row)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::Parameter>(element::f32, shapeB);

    auto bias = op::Constant::create<float>(element::f32, Shape{2}, std::vector<float>{2.0f, 3.0f});

    auto cg = make_shared<op::MatmulBias>(
        A, B, bias, A->get_shape(), B->get_shape(), true, true, AxisSet{0});

    auto f = make_shared<Function>(cg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    backend->call(f, {result}, {a, b});
    vector<float> expected{11, 30, 38, 111};
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(cpu_fusion, gemm_cpu_broadcast_column)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::Parameter>(element::f32, shapeB);

    auto bias = op::Constant::create<float>(element::f32, Shape{2}, std::vector<float>{2.0f, 3.0f});

    auto cg = make_shared<op::MatmulBias>(
        A, B, bias, A->get_shape(), B->get_shape(), true, true, AxisSet{1});

    auto f = make_shared<Function>(cg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    backend->call(f, {result}, {a, b});
    vector<float> expected{11, 29, 39, 111};
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(cpu_fusion, gemm_cpu_broadcast_matrix)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::Parameter>(element::f32, shapeB);

    auto reshape_w = make_shared<op::Reshape>(A, AxisVector{1, 0}, Shape{2, 3});
    auto reshape_x = make_shared<op::Reshape>(B, AxisVector{1, 0}, Shape{3, 2});

    auto one = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{1.0f});

    auto broadcast = make_shared<op::Broadcast>(one, shapeC, AxisSet{0, 1});
    auto cg = make_shared<op::MatmulBias>(
        A, B, one, A->get_shape(), B->get_shape(), true, true, AxisSet{0, 1});

    auto f = make_shared<Function>(cg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    backend->call(f, {result}, {a, b});
    vector<float> expected{10, 28, 37, 109};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

TEST(cpu_fusion, gemm_cpu_no_bias)
{
    auto shapeA = Shape{3, 2};
    auto shapeB = Shape{2, 3};
    auto shapeC = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::Parameter>(element::f32, shapeB);

    auto reshape_w = make_shared<op::Reshape>(A, AxisVector{1, 0}, Shape{2, 3});
    auto reshape_x = make_shared<op::Reshape>(B, AxisVector{1, 0}, Shape{3, 2});

    auto cg =
        make_shared<op::MatmulBias>(A, B, nullptr, A->get_shape(), B->get_shape(), true, true);

    auto f = make_shared<Function>(cg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    backend->call(f, {result}, {a, b});
    vector<float> expected{9, 27, 36, 108};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

TEST(cpu_fusion, cpu_fusion_pass_basic)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::Dot>(A, B);
    auto broadcast = make_shared<op::Broadcast>(C, dot->get_shape(), AxisSet{0});
    auto add = dot + broadcast;
    auto graph = make_shared<op::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::MatmulBias>(graph->get_input_op(0)), nullptr);
}

TEST(cpu_fusion, commutative_matmul_bias)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::Dot>(A, B);
    auto broadcast = make_shared<op::Broadcast>(C, dot->get_shape(), AxisSet{0});
    auto add = broadcast + dot;
    auto graph = make_shared<op::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::MatmulBias>(graph->get_input_op(0)), nullptr);
}

TEST(cpu_fusion, cpu_fusion_pass_matmul_bias)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto W = make_shared<op::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::Parameter>(element::f32, shape_x);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);

    auto mmb = std::make_shared<op::MatmulBias>(
        W, x, nullptr, W->get_shape(), x->get_shape(), false, false);
    auto broadcast = std::make_shared<op::Broadcast>(b, mmb->get_shape(), AxisSet{0});
    auto add = mmb + broadcast;

    auto graph = make_shared<op::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{W, x, b});
    pass_manager.run_passes(func);
    auto gmm = graph->get_input_op(0);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::MatmulBias>(gmm));
    ASSERT_EQ(gmm->get_input_op(2), b);
}

TEST(cpu_fusion, cpu_fusion_pass_matmul_no_bias)
{
    Shape shape_w{4, 2};
    Shape shape_x{1, 4};
    auto W = make_shared<op::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::Parameter>(element::f32, shape_x);

    auto reshape_w = std::make_shared<op::Reshape>(W, AxisVector{1, 0}, Shape{2, 4});
    auto reshape_x = std::make_shared<op::Reshape>(x, AxisVector{1, 0}, Shape{4, 1});
    auto re_dot = make_shared<op::Dot>(reshape_w, reshape_x);
    auto graph = make_shared<op::Abs>(re_dot);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{W, x});
    pass_manager.run_passes(func);
    size_t mmb = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(mmb, 1);
}

TEST(cpu_fusion, gemm_mlp)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/mnist_mlp_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    auto mmbs = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(mmbs, 3);
}

TEST(cpu_fusion, fuse_fprop_bn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_before_fusion.png");
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_after_fusion.png");
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop_b2c3h2w2.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::BatchNorm>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(cpu_fusion, zero_padded_reshaped_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 1});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, Shape{0, 1, 0, 0}, Shape{0, 0, 1, 0}, Shape{0, 0, 0, 0});

    auto reshape = make_shared<op::Reshape>(pad, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});

    auto conv = make_shared<op::Convolution>(reshape,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, op::ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(cpu_fusion, zero_padded_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, Shape{0, 0, 0, 1}, Shape{0, 0, 1, 0}, Shape{0, 0, 0, 0});

    auto conv = make_shared<op::Convolution>(pad,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, op::ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(cpu_fusion, non_zero_padded_conv)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{1.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, Shape{0, 0, 0, 1}, Shape{0, 0, 1, 0}, Shape{0, 0, 0, 0});

    auto conv = make_shared<op::Convolution>(pad,
                                             F,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});

    auto func = make_shared<Function>(conv, op::ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);
}

TEST(cpu_fusion, zero_padded_conv_backprop_filters)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});

    auto pad_value = op::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad =
        make_shared<op::Pad>(X, pad_value, Shape{0, 0, 0, 1}, Shape{0, 0, 1, 0}, Shape{0, 0, 0, 0});

    auto conv = make_shared<op::ConvolutionBackpropFilters>(pad,
                                                            Shape{1, 1, 2, 2},
                                                            F,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});

    auto func = make_shared<Function>(conv, op::ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 1);

    auto backend = runtime::Backend::create("CPU");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::Pad>(func), 0);
}

TEST(cpu_fusion, fuse_conv_bias)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "conv_bias.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t cb = count_ops_of_type<op::ConvolutionBias>(func);
    ASSERT_GT(cb, 0);
}

struct ConvolutionBiasTestData
{
    size_t n{0};
    size_t c{0};
    size_t filter{0};
    size_t kernel_size{0};
    size_t w{0};
    size_t h{0};
    shared_ptr<runtime::TensorView> data_val;
    shared_ptr<runtime::TensorView> weights_val;
    shared_ptr<runtime::TensorView> bias_val;
    shared_ptr<runtime::TensorView> result_val;
    shared_ptr<runtime::TensorView> delta_val;
    shared_ptr<runtime::TensorView> d_data_val;
    shared_ptr<runtime::TensorView> d_weights_val;
    shared_ptr<runtime::TensorView> d_bias_val;
    vector<float> expected_result_val;
    vector<float> expected_d_data_val;
    vector<float> expected_d_weights_val;
    vector<float> expected_d_bias_val;

    Shape data_shape;
    Shape weights_shape;
    Shape bias_shape;
    Shape result_shape;
    shared_ptr<op::Parameter> data;
    shared_ptr<op::Parameter> weights;
    shared_ptr<op::Parameter> bias;
    shared_ptr<op::Parameter> delta;

    void n1c1h3w3(shared_ptr<runtime::Backend> backend)
    {
        n = 1;
        c = 1;
        filter = 1;
        kernel_size = 3;
        w = 3;
        h = w;

        data_shape = Shape{n, c, h, w};
        data = make_shared<op::Parameter>(element::f32, data_shape);
        weights_shape = Shape{filter, c, kernel_size, kernel_size};
        weights = make_shared<op::Parameter>(element::f32, weights_shape);
        bias_shape = Shape{filter};
        bias = make_shared<op::Parameter>(element::f32, bias_shape);
        result_shape = Shape{n, filter, 1, 1};

        data_val = backend->create_tensor(element::f32, data_shape);
        copy_data(data_val,
                  vector<float>{-0.67765152f,
                                0.10073948f,
                                0.57595438f,
                                -0.3469252f,
                                -0.22134334f,
                                -1.80471897f,
                                -0.80642909f,
                                1.22033095f,
                                2.23235631f});
        weights_val = backend->create_tensor(element::f32, weights_shape);
        copy_data(weights_val,
                  vector<float>{0.20070229f,
                                -0.54968649f,
                                -0.19819015f,
                                -0.38577855f,
                                1.37109005f,
                                -0.23789984f,
                                0.14867957f,
                                -0.49851316f,
                                -0.84815776f});
        bias_val = backend->create_tensor(element::f32, bias_shape);
        copy_data(bias_val, vector<float>{0.07811152f});

        result_val = backend->create_tensor(element::f32, result_shape);
        copy_data(result_val, vector<float>{0});

        delta = make_shared<op::Parameter>(element::f32, result_shape);
        delta_val = backend->create_tensor(element::f32, result_shape);
        copy_data(delta_val, vector<float>{-2.58936238f});

        d_data_val = backend->create_tensor(element::f32, data_shape);
        copy_data(d_data_val, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0});

        d_weights_val = backend->create_tensor(element::f32, weights_shape);
        copy_data(d_weights_val, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0});

        d_bias_val = backend->create_tensor(element::f32, bias_shape);
        copy_data(d_bias_val, vector<float>{0});

        expected_result_val = vector<float>{-2.58936238f};
        expected_d_data_val = vector<float>{-0.51969099f,
                                            1.42333758f,
                                            0.5131861f,
                                            0.99892044f,
                                            -3.5502491f,
                                            0.61600888f,
                                            -0.3849853f,
                                            1.29083121f,
                                            2.19618773f};
        expected_d_weights_val = vector<float>{1.7546854f,
                                               -0.26085103f,
                                               -1.49135458f,
                                               0.89831507f,
                                               0.57313812f,
                                               4.67307138f,
                                               2.08813715f,
                                               -3.15987897f,
                                               -5.7803793f};
        expected_d_bias_val = vector<float>{-2.58936238f};
    }
};

TEST(cpu_fusion, conv_bias_fprop_n1c1h3w3)
{
    auto backend = runtime::Backend::create("CPU");

    ConvolutionBiasTestData conv_test;
    conv_test.n1c1h3w3(backend);

    auto convolution = make_shared<op::Convolution>(conv_test.data, conv_test.weights);
    auto convolution_bias = make_shared<op::ConvolutionBias>(convolution, conv_test.bias);

    auto f = make_shared<Function>(
        convolution_bias, op::ParameterVector{conv_test.data, conv_test.weights, conv_test.bias});

    backend->call(
        f, {conv_test.result_val}, {conv_test.data_val, conv_test.weights_val, conv_test.bias_val});
    auto result_vec = read_vector<float>(conv_test.result_val);

    EXPECT_TRUE(
        test::all_close(conv_test.expected_result_val, read_vector<float>(conv_test.result_val)));
}

TEST(cpu_fusion, conv_bias_bprop_n1c1h3w3)
{
    auto backend = runtime::Backend::create("CPU");

    ConvolutionBiasTestData conv_test;
    conv_test.n1c1h3w3(backend);

    auto convolution = make_shared<op::Convolution>(conv_test.data, conv_test.weights);
    auto convolution_bias = make_shared<op::ConvolutionBias>(convolution, conv_test.bias);

    auto f = make_shared<Function>(
        convolution_bias, op::ParameterVector{conv_test.data, conv_test.weights, conv_test.bias});

    ngraph::autodiff::Adjoints adjoints(NodeVector{convolution_bias}, NodeVector{conv_test.delta});

    auto d_data = adjoints.backprop_node(conv_test.data);
    auto d_weights = adjoints.backprop_node(conv_test.weights);
    auto d_bias = adjoints.backprop_node(conv_test.bias);

    auto df = make_shared<Function>(
        NodeVector{d_data, d_weights, d_bias},
        op::ParameterVector{conv_test.data, conv_test.weights, conv_test.bias, conv_test.delta});

    backend->call(
        df,
        {conv_test.d_data_val, conv_test.d_weights_val, conv_test.d_bias_val},
        {conv_test.data_val, conv_test.weights_val, conv_test.bias_val, conv_test.delta_val});

    EXPECT_TRUE(
        test::all_close(conv_test.expected_d_data_val, read_vector<float>(conv_test.d_data_val)));
    EXPECT_TRUE(test::all_close(conv_test.expected_d_weights_val,
                                read_vector<float>(conv_test.d_weights_val)));
    EXPECT_TRUE(
        test::all_close(conv_test.expected_d_bias_val, read_vector<float>(conv_test.d_bias_val)));
}

TEST(cpu_fusion, sigmoid_fprop_fusion)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::Sigmoid>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(cpu_fusion, sigmoid_n1c1h2w2)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::TensorView> result =
        backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    backend->call(func, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

TEST(cpu_fusion, sigmoid_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::TensorView> result =
        backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    backend->call(func, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

TEST(cpu_fusion, sigmoid_bprop_fusion)
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

TEST(cpu_fusion, sigmoid_bprop_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::SigmoidBackprop>(input, delta);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input, delta});
    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, delta->get_shape());
    shared_ptr<runtime::TensorView> result =
        backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{1.0f, 1.0f, 1.0f, 1.0f};

    copy_data(a, dataA);
    copy_data(b, dataB);
    backend->call(func, {result}, {a, b});

    vector<float> expected{0.196612f, 0.0176627f, 0.196612f, 0.0176627f};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

TEST(cpu_fusion, batchnorm_fprop_relu_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    // Note, op::Splice is used to break Relu(BatchNorm) fusion
    // otherwise we will be comparing two BatchNormRelus
    // Unfortunately, we can't use INTERPRETER for
    // verifying the results as it doesn't implement
    // BatchNorm op.
    auto slice =
        std::make_shared<op::Slice>(output_rt, Coordinate{0, 0, 0, 0}, Coordinate{1, 2, 2, 2});
    auto output_relu = std::make_shared<op::Relu>(slice);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto bn_relu = make_shared<op::BatchNormRelu>(eps, gamma, beta, input);
    auto output_rt_bnr = std::make_shared<op::GetOutputElement>(bn_relu, 0);
    auto mean_rt_bnr = std::make_shared<op::GetOutputElement>(bn_relu, 1);
    auto variance_rt_bnr = std::make_shared<op::GetOutputElement>(bn_relu, 2);

    auto f = make_shared<Function>(
        NodeVector{output_relu, mean_rt, variance_rt, output_rt_bnr, mean_rt_bnr, variance_rt_bnr},
        op::ParameterVector{input, gamma, beta});
    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto input_t = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

    copy_data(input_t,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto gamma_t = backend->create_tensor(element::f32, gamma_shape);
    copy_data(gamma_t, vector<float>{1.0f, 1.0f});
    auto beta_t = backend->create_tensor(element::f32, beta_shape);
    copy_data(beta_t, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    auto bn_output_bnr = backend->create_tensor(element::f32, shape_r);
    auto result_mean_bnr = backend->create_tensor(element::f32, mean_shape);
    auto result_variance_bnr = backend->create_tensor(element::f32, var_shape);

    backend->call(f,
                  {bn_output,
                   result_mean,
                   result_variance,
                   bn_output_bnr,
                   result_mean_bnr,
                   result_variance_bnr},
                  {input_t, gamma_t, beta_t});

    EXPECT_TRUE(test::all_close(read_vector<float>(bn_output), read_vector<float>(bn_output_bnr)));
    EXPECT_TRUE(
        test::all_close(read_vector<float>(result_mean), read_vector<float>(result_mean_bnr)));
    EXPECT_TRUE(test::all_close(read_vector<float>(result_variance),
                                read_vector<float>(result_variance_bnr)));
}

TEST(cpu_fusion, fuse_conv_relu)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = std::make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto convolution = std::make_shared<op::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto relu = std::make_shared<op::Relu>(convolution);
    auto abs_node =
        std::make_shared<op::Abs>(std::make_shared<op::Abs>(std::make_shared<op::Abs>(relu)));
    auto func = make_shared<Function>(abs_node, op::ParameterVector{A, weights});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    size_t cb = count_ops_of_type<op::ConvolutionRelu>(func);
    ASSERT_GT(cb, 0);
}

template <typename T>
static std::vector<std::vector<T>>
    execute(std::shared_ptr<Function> f, std::vector<std::vector<T>> args, std::string cbackend)
{
    auto backend = runtime::Backend::create("CPU");

    auto parms = f->get_parameters();

    if (parms.size() != args.size())
    {
        throw ngraph_error("number of parameters and arguments don't match");
    }

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> arg_tensors(args.size());
    for (size_t i = 0; i < args.size(); i++)
    {
        auto t = backend->create_tensor(parms.at(i)->get_element_type(), parms.at(i)->get_shape());
        copy_data(t, args.at(i));
        arg_tensors.at(i) = t;
    }

    auto results = f->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> result_tensors(results.size());

    for (size_t i = 0; i < results.size(); i++)
    {
        result_tensors.at(i) =
            backend->create_tensor(results.at(i)->get_element_type(), results.at(i)->get_shape());
    }

    backend->call(f, result_tensors, arg_tensors);

    std::vector<std::vector<T>> result_vectors;
    for (auto rt : result_tensors)
    {
        result_vectors.push_back(read_vector<T>(rt));
    }
    return result_vectors;
}

TEST(cpu_fusion, conv_relu_n2c1h2w2_2)
{
    Shape shape_a{2, 1, 6, 6};
    Shape shape_weights{1, 1, 2, 2};

    auto make_int_function = [shape_a, shape_weights]() {
        auto A = std::make_shared<op::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto relu = std::make_shared<op::Relu>(conv);
        auto f = make_shared<Function>(NodeVector{relu}, op::ParameterVector{A, weights});
        return f;
    };

    auto int_f = make_int_function();

    auto make_cpu_function = [shape_a, shape_weights]() {
        auto A = std::make_shared<op::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto conv_relu = std::make_shared<op::ConvolutionRelu>(conv);
        auto f = make_shared<Function>(NodeVector{conv_relu}, op::ParameterVector{A, weights});
        return f;
    };

    auto cpu_f = make_cpu_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {2., 2., 2., 2.}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

std::vector<shared_ptr<runtime::TensorView>>
    rnn_matrix_fusion_eval(const size_t time_steps,
                           const Shape& data_shape,
                           const Shape& weights_shape,
                           const Shape& bias_shape,
                           const vector<float>& data_val,
                           const vector<float>& weights_val,
                           const vector<float>& bias_val,
                           const bool enable_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto weights = make_shared<op::Parameter>(element::f32, weights_shape);
    auto bias = make_shared<op::Parameter>(element::f32, bias_shape);

    // results from each time step
    NodeVector results;
    for (size_t t = 0; t < time_steps; ++t)
    {
        auto data_slice = make_shared<op::Slice>(
            data, Coordinate{0, t, 0}, Coordinate{data_shape[0], t + 1, data_shape[2]});
        auto data_reshape = make_shared<op::Reshape>(
            data_slice, AxisVector{0, 1, 2}, Shape{data_shape[0], data_shape[2]});
        auto weights_reshape = make_shared<op::Reshape>(
            weights, AxisVector{1, 0}, Shape{weights_shape[1], weights_shape[0]});
        auto dot = make_shared<op::Dot>(data_reshape, weights_reshape);
        auto bias_broadcast = make_shared<op::Broadcast>(bias, dot->get_shape(), AxisSet{0});
        auto add = make_shared<op::Add>(dot, bias_broadcast);
        results.push_back(add);
    }
    auto func = make_shared<Function>(results, op::ParameterVector{data, weights, bias});
    if (enable_pass)
    {
        pass::Manager pass_manager;
        pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
        pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
        pass_manager.run_passes(func);
        // check all of our dot/add are converted to a single MatmulBias op.
        size_t count = count_ops_of_type<op::MatmulBias>(func);
        EXPECT_EQ(count, 1);
    }

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> data_tensor =
        backend->create_tensor(element::f32, data->get_shape());
    shared_ptr<runtime::TensorView> weights_tensor =
        backend->create_tensor(element::f32, weights->get_shape());
    shared_ptr<runtime::TensorView> bias_tensor =
        backend->create_tensor(element::f32, bias->get_shape());

    std::vector<shared_ptr<runtime::TensorView>> result_tensors;
    for (auto r : results)
    {
        result_tensors.push_back(backend->create_tensor(element::f32, r->get_shape()));
    }

    copy_data(data_tensor, data_val);
    copy_data(weights_tensor, weights_val);
    copy_data(bias_tensor, bias_val);
    backend->call(func, result_tensors, {data_tensor, weights_tensor, bias_tensor});
    return result_tensors;
}

TEST(cpu_fusion, rnn_matrix_fusion_eval_pass)
{
    const size_t time_steps = 4;
    Shape data_shape{3, time_steps, 5};
    Shape weights_shape{6, data_shape[2]};
    Shape bias_shape{6};

    test::Uniform<float> rng{0, 1, 0};
    vector<float> data_val(shape_size(data_shape));
    vector<float> weights_val(shape_size(weights_shape));
    vector<float> bias_val(shape_size(bias_shape));
    rng.initialize(data_val);
    rng.initialize(weights_val);
    rng.initialize(bias_val);

    std::vector<shared_ptr<runtime::TensorView>> result_expected = rnn_matrix_fusion_eval(
        time_steps, data_shape, weights_shape, bias_shape, data_val, weights_val, bias_val, false);
    std::vector<shared_ptr<runtime::TensorView>> result_fused = rnn_matrix_fusion_eval(
        time_steps, data_shape, weights_shape, bias_shape, data_val, weights_val, bias_val, true);
    for (size_t i = 0; i < result_expected.size(); ++i)
    {
        EXPECT_TRUE(test::all_close<float>(result_expected[i], result_fused[i]));
    }
}

TEST(cpu_fusion, rnn_fusion_from_json_model)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/rnn-10-step-fusion-test.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);

    const size_t NUM_STEPS = 10;
    auto mmb_predicate = [](std::shared_ptr<Node> node) {
        auto users = node->get_users();
        return users.size() == NUM_STEPS &&
               std::all_of(begin(users), end(users), [](std::shared_ptr<Node> n) {
                   return std::dynamic_pointer_cast<op::Slice>(n) != nullptr;
               });
    };

    auto mmbs = get_ops_of_type<op::MatmulBias>(func);
    ASSERT_TRUE(std::any_of(begin(mmbs), end(mmbs), mmb_predicate));
}
