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
#include "ngraph/op/concat.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/op/batch_dot.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/pass/cpu_concat_inputs.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_loop_kernel_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

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

    auto skip_w = std::make_shared<pattern::op::Skip>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Skip>(x, reshape_pred);

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

    backend->call_with_validate(f, {result}, {a, b});
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

    backend->call_with_validate(f, {result}, {a, b});
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

    backend->call_with_validate(f, {result}, {a, b});
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

    backend->call_with_validate(f, {result}, {a, b});
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, op::ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::MatmulBias>(graph->get_argument(0)), nullptr);
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, op::ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::MatmulBias>(graph->get_argument(0)), nullptr);
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, op::ParameterVector{W, x, b});
    pass_manager.run_passes(func);
    auto gmm = graph->get_argument(0);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::MatmulBias>(gmm));
    ASSERT_EQ(gmm->get_argument(2), b);
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
    pass_manager.run_passes(func);
    auto mmbs = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(mmbs, 3);
}

TEST(cpu_fusion, fuse_fprop_bn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_before_fusion.png");
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::DIFFERENTIABLE_FUSIONS);
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

    backend->call_with_validate(
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
    backend->call_with_validate(
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

TEST(cpu_fusion, conv_bias_bprop)
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<op::Parameter>(element::f32, shape);
    auto filters = std::make_shared<op::Parameter>(element::f32, shape);
    auto delta = std::make_shared<op::Parameter>(element::f32, shape);
    auto bias = make_shared<op::Parameter>(element::f32, Shape{shape[0]});
    auto pbroadcast = std::make_shared<op::Broadcast>(bias, shape, AxisSet{1, 2, 3});
    auto conv = std::make_shared<op::Convolution>(data_batch, filters);
    auto conv_bias = std::make_shared<op::Add>(conv, pbroadcast);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("conv_bias_bprop_fusion");
    auto f = make_shared<Function>(conv_bias, op::ParameterVector{data_batch, filters, bias});

    ngraph::autodiff::Adjoints adjoints(NodeVector{conv_bias}, NodeVector{delta});

    auto d_data = adjoints.backprop_node(data_batch);
    auto d_weights = adjoints.backprop_node(filters);
    auto d_bias = adjoints.backprop_node(bias);

    auto df = make_shared<Function>(NodeVector{d_data, d_weights, d_bias},
                                    op::ParameterVector{data_batch, filters, bias, delta});

    pass_manager.run_passes(df);
    size_t ccg = count_ops_of_type<op::ConvolutionBiasBackpropFiltersBias>(df);
    ASSERT_EQ(ccg, 1);
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

    backend->call_with_validate(f,
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
    pass_manager.run_passes(func);
    size_t cb = count_ops_of_type<op::ConvolutionRelu>(func);
    ASSERT_GT(cb, 0);
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

TEST(cpu_fusion, conv_bias_relu_n2c1h2w2_2)
{
    Shape shape_a{2, 1, 6, 6};
    Shape shape_weights{1, 1, 2, 2};
    Shape shape_bias{1};

    auto make_int_function = [shape_a, shape_weights, shape_bias]() {
        auto A = std::make_shared<op::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto bias = std::make_shared<op::Parameter>(element::f32, shape_bias);
        auto conv_bias =
            conv + std::make_shared<op::Broadcast>(bias, conv->get_shape(), AxisSet{0, 2, 3});
        auto relu = std::make_shared<op::Relu>(conv_bias);
        auto f = make_shared<Function>(NodeVector{relu}, op::ParameterVector{A, weights, bias});
        return f;
    };

    auto int_f = make_int_function();

    auto make_cpu_function = [shape_a, shape_weights, shape_bias]() {
        auto A = std::make_shared<op::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::Parameter>(element::f32, shape_bias);
        auto conv = std::make_shared<op::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto conv_bias_relu = std::make_shared<op::ConvolutionBias>(conv, bias, true);
        auto f = make_shared<Function>(NodeVector{conv_bias_relu},
                                       op::ParameterVector{A, weights, bias});
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
        {2., 2., 2., 2.},
        {0.1f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

// ConvolutionBiasAdd relies on an in-place fused MKLDNN kernel.
// Need to ensure that it is fused only when in-place buffer allocation is feasible
shared_ptr<Function> gen_conv_bias_add(bool param_input, bool result_output)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{1});
    auto conv = make_shared<op::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto bias_broadcast = make_shared<op::Broadcast>(bias, conv->get_shape(), AxisSet{0, 2, 3});
    auto convbias = conv + bias_broadcast;
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto abs_B = make_shared<op::Abs>(B);
    auto add =
        param_input ? make_shared<op::Add>(convbias, B) : make_shared<op::Add>(convbias, abs_B);
    auto abs = make_shared<op::Abs>(add);

    return result_output ? make_shared<Function>(add, op::ParameterVector{A, weights, bias, B})
                         : make_shared<Function>(abs, op::ParameterVector{A, weights, bias, B});
}

TEST(cpu_fusion, fuse_conv_bias_add)
{
    auto func_fuse = gen_conv_bias_add(false, false);
    auto func_nofuse1 = gen_conv_bias_add(true, false);
    auto func_nofuse2 = gen_conv_bias_add(false, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func_fuse);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBiasAdd>(func_fuse), 1);

    pass_manager.run_passes(func_nofuse1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBiasAdd>(func_nofuse1), 0);

    pass_manager.run_passes(func_nofuse2);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionBiasAdd>(func_nofuse2), 1);
}

TEST(cpu_fusion, conv_bias_add)
{
    auto int_f = gen_conv_bias_add(false, false);
    auto cpu_f = gen_conv_bias_add(false, false);

    vector<vector<float>> args{{1.25f, 2.25f, 5.25f, 6.25f, -1.25f, -1.25f, 3.25f, -4.25f},
                               {-1.25f},
                               {2.25f},
                               {1.25f, 2.25f, -3.25f, 2.25f, 4.25f, 4.25f, 1.25f, 2.25f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

// ConvolutionAdd relies on an in-place fused MKLDNN kernel.
// Need to ensure that it is fused only when in-place buffer allocation is feasible
shared_ptr<Function> gen_conv_add(bool param_input, bool result_output)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1});
    auto conv = make_shared<op::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto abs_B = make_shared<op::Abs>(B);
    auto add = param_input ? make_shared<op::Add>(conv, B) : make_shared<op::Add>(conv, abs_B);
    auto abs = make_shared<op::Abs>(add);

    return result_output ? make_shared<Function>(add, op::ParameterVector{A, weights, B})
                         : make_shared<Function>(abs, op::ParameterVector{A, weights, B});
}

TEST(cpu_fusion, fuse_conv_add)
{
    auto func_fuse = gen_conv_add(false, false);
    auto func_nofuse1 = gen_conv_add(true, false);
    auto func_nofuse2 = gen_conv_add(false, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func_fuse);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_fuse), 1);

    pass_manager.run_passes(func_nofuse1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_nofuse1), 0);

    pass_manager.run_passes(func_nofuse2);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_nofuse2), 1);
}

TEST(cpu_fusion, conv_add)
{
    auto int_f = gen_conv_add(false, false);
    auto cpu_f = gen_conv_add(false, false);

    vector<vector<float>> args{{1.25f, 2.25f, 5.25f, 6.25f, -1.25f, -1.25f, 3.25f, -4.25f},
                               {-1.25f},
                               {1.25f, 2.25f, -3.25f, 2.25f, 4.25f, 4.25f, 1.25f, 2.25f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));

    int_f = gen_conv_add(false, true);
    cpu_f = gen_conv_add(false, true);

    int_results = execute(int_f, args, "INTERPRETER");
    cpu_results = execute(cpu_f, args, "CPU");
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
        pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
            runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
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
    backend->call_with_validate(func, result_tensors, {data_tensor, weights_tensor, bias_tensor});
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
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
        runtime::cpu::pass::CPUFusion::REGULAR_FUSIONS);
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

TEST(cpu_fusion, weight_fusion)
{
    auto param = std::make_shared<op::Parameter>(element::f32, Shape{64});
    auto reshape_conv =
        std::make_shared<ngraph::op::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto data_conv = std::make_shared<op::Parameter>(element::f32, Shape{16, 4, 7, 7});
    auto tvt = reshape_conv->get_outputs().at(0).get_tensor_ptr().get();
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt);
    auto cvt_lt_conv = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv, lt_desc);
    auto conv = std::make_shared<ngraph::op::Convolution>(
        data_conv, cvt_lt_conv, Strides{1, 1}, Strides{1, 1});

    auto reshape_conv_bprop =
        std::make_shared<op::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto dummy_arg_conv_bprop = std::make_shared<op::Parameter>(element::f32, Shape{1, 16, 7, 7});
    auto tvt_bprop = reshape_conv_bprop->get_outputs().at(0).get_tensor_ptr().get();
    auto lt_desc_bprop = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt_bprop);
    auto cvt_lt_conv_bprop =
        std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv_bprop, lt_desc_bprop);
    auto conv_bprop = std::make_shared<op::ConvolutionBackpropData>(Shape{1, 4, 7, 7},
                                                                    cvt_lt_conv_bprop,
                                                                    dummy_arg_conv_bprop,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1});

    auto conv_relu = std::make_shared<op::Relu>(conv);
    auto conv_bprop_abs = std::make_shared<op::Abs>(conv_bprop);

    auto f = make_shared<Function>(NodeVector{conv_relu, conv_bprop_abs},
                                   op::ParameterVector{param, data_conv, dummy_arg_conv_bprop});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUPostLayoutOptimizations>();
    pass_manager.run_passes(f);

    auto new_conv_bprop_data = conv_bprop_abs->get_argument(0);
    auto new_convert_layout = new_conv_bprop_data->get_argument(0);

    ASSERT_EQ(std::dynamic_pointer_cast<runtime::cpu::op::ConvertLayout>(
                  new_convert_layout->get_argument(0)),
              cvt_lt_conv);
}

TEST(cpu_fusion, max_pool_with_indices)
{
    Shape shape_a{10, 3, 28, 28};
    auto input = std::make_shared<op::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto max_pool = std::make_shared<op::MaxPool>(input, window_shape);
    auto C = std::make_shared<op::Parameter>(element::f32, max_pool->get_shape());

    ngraph::autodiff::Adjoints adjoints(NodeVector{max_pool}, NodeVector{C});

    auto dinput = adjoints.backprop_node(input);

    auto df = std::make_shared<Function>(NodeVector{dinput}, op::ParameterVector{input, C});

    auto f = std::make_shared<Function>(NodeVector{max_pool}, op::ParameterVector{input});

    {
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_fprop_before.pdf");
        pass_manager.run_passes(f);
    }

    {
        NodeVector nv_cwi;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_bprop_before.pdf");
        pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_bprop_after.pdf");
        pass_manager.run_passes(df);
    }

    {
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_fprop_after.pdf");
        pass_manager.run_passes(f);
    }

    auto maxpool_goe_output =
        std::dynamic_pointer_cast<op::GetOutputElement>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(maxpool_goe_output);
    ASSERT_EQ(maxpool_goe_output->get_n(), 0);
    auto maxpool_with_indices = df->get_results().at(0)->get_argument(0);
    auto maxpool_goe_indices =
        std::dynamic_pointer_cast<op::GetOutputElement>(maxpool_with_indices->get_argument(2));
    ASSERT_TRUE(maxpool_goe_indices);
    ASSERT_EQ(maxpool_goe_indices->get_n(), 1);
}

TEST(cpu_fusion, backwards_maxpool_with_indices_n4_c1_hw4_2x2_max)
{
    Shape shape_a{1, 4, 4, 4};
    Shape maxpool_shape{1, 4, 3, 3};
    auto A = std::make_shared<op::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto maxpool = std::make_shared<op::MaxPool>(A, window_shape, window_movement_strides);
    auto f = std::make_shared<Function>(maxpool, op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");
    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::f32, maxpool_shape);
    vector<float> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::f32, shape_a);
    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::f32, shape_a);

    vector<float> dataInput{11.f, 31.f, 40.f, 47.f, 13.f, 61.f, 48.f, 59.f, 17.f, 39.f, 64.f,
                            62.f, 45.f, 55.f, 36.f, 19.f, 65.f, 33.f, 49.f, 30.f, 56.f, 41.f,
                            53.f, 58.f, 22.f, 35.f, 52.f, 50.f, 63.f, 54.f, 12.f, 26.f, 44.f,
                            21.f, 69.f, 24.f, 46.f, 25.f, 51.f, 29.f, 72.f, 15.f, 73.f, 10.f,
                            16.f, 37.f, 70.f, 32.f, 28.f, 66.f, 57.f, 27.f, 60.f, 42.f, 43.f,
                            71.f, 18.f, 38.f, 67.f, 68.f, 14.f, 20.f, 34.f, 23.f};

    vector<float> expected{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 12.0f, 0.0f, 4.0f, 0.0f, 0.0f,  16.0f,
                           0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 4.0f,  0.0f, 0.0f, 0.0f, 4.0f,  0.0f,
                           8.0f, 8.0f, 0.0f, 0.0f, 4.0f, 0.0f,  4.0f, 4.0f, 0.0f, 0.0f,  0.0f,
                           0.0f, 8.0f, 0.0f, 4.0f, 0.0f, 0.0f,  0.0f, 8.0f, 0.0f, 16.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f,  0.0f, 0.0f, 4.0f, 0.0f,  0.0f,
                           8.0f, 0.0f, 4.0f, 8.0f, 4.0f, 0.0f,  0.0f, 0.0f, 0.0f};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = std::make_shared<op::Parameter>(element::f32, maxpool_shape);
    auto df = autodiff::backprop_function(f);

    {
        NodeVector nv_cwi;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_bprop_before2.pdf");
        pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
        pass_manager.register_pass<pass::VisualizeTree>("max_pool_bprop_after2.pdf");
        pass_manager.run_passes(df);
    }

    backend->call_with_validate(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<float>(output) == expected);
}

TEST(cpu_fusion, loop_kernel_one_input_one_output)
{
    Shape shapeA{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shapeA);
    auto neg_a = make_shared<op::Negative>(A);
    auto lk = make_shared<runtime::cpu::op::LoopKernel>(
        NodeVector{neg_a}, NodeVector{neg_a}, NodeVector{A});
    auto f = make_shared<Function>(NodeVector{lk}, op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shapeA);

    vector<int> dataA{1, 4, 1, 4};
    copy_data(a, dataA);
    vector<int> expected{-1, -4, -1, -4};

    backend->call_with_validate(f, {result}, {a});

    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_fusion, loop_kernel_embedded_graph)
{
    Shape shapeA{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shapeA);
    auto B = make_shared<op::Parameter>(element::i32, shapeA);
    auto neg_a = make_shared<op::Negative>(A);
    auto neg_b = make_shared<op::Negative>(B);
    auto add = neg_a + neg_b;
    auto lk = make_shared<runtime::cpu::op::LoopKernel>(
        NodeVector{add}, NodeVector{add}, NodeVector{neg_a, neg_b});
    auto f = make_shared<Function>(NodeVector{lk}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shapeA);

    vector<int> dataA{1, 4, 1, 4};
    copy_data(a, dataA);
    vector<int> dataB{1, 2, 3, 4};
    copy_data(b, dataB);
    vector<int> expected{-2, -6, -4, -8};
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_fusion, loop_kernel_two_inputs_one_output)
{
    Shape shapeA{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shapeA);
    auto B = make_shared<op::Parameter>(element::i32, shapeA);
    auto add = A + B;
    auto lk = make_shared<runtime::cpu::op::LoopKernel>(
        NodeVector{add}, NodeVector{add}, NodeVector{A, B});
    auto f = make_shared<Function>(NodeVector{lk}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shapeA);

    vector<int> dataA{1, 4, 1, 4};
    copy_data(a, dataA);
    vector<int> dataB{1, 2, 3, 4};
    copy_data(b, dataB);
    vector<int> expected{2, 6, 4, 8};

    backend->call_with_validate(f, {result}, {a, b});

    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_fusion, loop_kernel_multiple_outputs)
{
    Shape shapeA{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shapeA);
    auto B = make_shared<op::Parameter>(element::i32, shapeA);
    auto C = make_shared<op::Parameter>(element::i32, shapeA);
    auto D = make_shared<op::Parameter>(element::i32, shapeA);

    auto neg_a = make_shared<op::Negative>(A);
    auto neg_b = make_shared<op::Negative>(B);
    auto add_ab = neg_a + neg_b;
    auto add_cd = C + B;
    auto add_cd_abs = make_shared<op::Abs>(add_cd);
    auto add_ab_abs = make_shared<op::Abs>(add_ab);
    auto add_aab = add_ab_abs + A;
    auto add_cdd = add_cd_abs + D;

    auto lk = make_shared<runtime::cpu::op::LoopKernel>(
        NodeVector{neg_a, neg_b, add_ab, add_cd, add_cd_abs, add_ab_abs, add_aab, add_cdd},
        NodeVector{add_aab, add_cdd, neg_b},
        NodeVector{A, B, C, D});
    auto add_aab_goe = std::make_shared<op::GetOutputElement>(lk, 0);
    auto add_cdd_goe = std::make_shared<op::GetOutputElement>(lk, 1);
    auto neg_b_goe = std::make_shared<op::GetOutputElement>(lk, 2);

    auto f = make_shared<Function>(NodeVector{add_aab_goe, add_cdd_goe, neg_b_goe},
                                   op::ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> d = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r1 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r2 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r3 = backend->create_tensor(element::i32, shapeA);

    vector<int> dataA{1, 4, 1, 4};
    vector<int> dataB{3, 3, 3, 9};
    vector<int> dataC{1, 2, 3, 4};
    vector<int> dataD{-2, 2, -1, 1};
    copy_data(a, dataA);
    copy_data(b, dataB);
    copy_data(c, dataC);
    copy_data(d, dataD);

    backend->call_with_validate(f, {r1, r2, r3}, {a, b, c, d});

    vector<int> expected1{5, 11, 5, 17};
    vector<int> expected2{2, 7, 5, 14};
    vector<int> expected3{-3, -3, -3, -9};
    EXPECT_EQ(read_vector<int>(r1), expected1);
    EXPECT_EQ(read_vector<int>(r2), expected2);
    EXPECT_EQ(read_vector<int>(r3), expected3);
}

TEST(cpu_fusion, loop_kernel_copy_with_new_args)
{
    Shape shapeA{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shapeA);
    auto B = make_shared<op::Parameter>(element::i32, shapeA);
    auto C = make_shared<op::Parameter>(element::i32, shapeA);
    auto D = make_shared<op::Parameter>(element::i32, shapeA);

    auto neg_a = make_shared<op::Negative>(A);
    auto neg_b = make_shared<op::Negative>(B);
    auto add_ab = neg_a + neg_b;
    auto add_cd = C + B;
    auto add_cd_abs = make_shared<op::Abs>(add_cd);
    auto add_ab_abs = make_shared<op::Abs>(add_ab);
    auto add_aab = add_ab_abs + A;
    auto add_cdd = add_cd_abs + D;

    auto lk = make_shared<runtime::cpu::op::LoopKernel>(
        NodeVector{neg_a, neg_b, add_ab, add_cd, add_cd_abs, add_ab_abs, add_aab, add_cdd},
        NodeVector{add_aab, add_cdd, neg_b},
        NodeVector{A, B, C, D});
    auto add_aab_goe = std::make_shared<op::GetOutputElement>(lk, 0);
    auto add_cdd_goe = std::make_shared<op::GetOutputElement>(lk, 1);
    auto neg_b_goe = std::make_shared<op::GetOutputElement>(lk, 2);

    auto f = make_shared<Function>(NodeVector{add_aab_goe, add_cdd_goe, neg_b_goe},
                                   op::ParameterVector{A, B, C, D});

    auto copy_f = clone_function(*f);

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> d = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r1 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r2 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> r3 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> copy_r1 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> copy_r2 = backend->create_tensor(element::i32, shapeA);
    shared_ptr<runtime::TensorView> copy_r3 = backend->create_tensor(element::i32, shapeA);

    vector<int> dataA{1, 4, 1, 4};
    vector<int> dataB{3, 3, 3, 9};
    vector<int> dataC{1, 2, 3, 4};
    vector<int> dataD{-2, 2, -1, 1};
    copy_data(a, dataA);
    copy_data(b, dataB);
    copy_data(c, dataC);
    copy_data(d, dataD);

    backend->call_with_validate(f, {r1, r2, r3}, {a, b, c, d});
    backend->call_with_validate(copy_f, {copy_r1, copy_r2, copy_r3}, {a, b, c, d});

    EXPECT_EQ(read_vector<int>(r1), read_vector<int>(copy_r1));
    EXPECT_EQ(read_vector<int>(r2), read_vector<int>(copy_r2));
    EXPECT_EQ(read_vector<int>(r3), read_vector<int>(copy_r3));
}
static std::shared_ptr<ngraph::Function> make_forward_function()
{
    Shape shape_a{10, 3, 28, 28};
    auto input = std::make_shared<op::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto max_pool = std::make_shared<op::MaxPool>(input, window_shape);
    auto neg = std::make_shared<op::Negative>(max_pool);
    auto absn = std::make_shared<op::Abs>(max_pool);
    return std::make_shared<Function>(NodeVector{max_pool, neg, absn}, op::ParameterVector{input});
}

static std::pair<std::shared_ptr<ngraph::Function>, std::vector<std::shared_ptr<ngraph::Node>>>
    make_backward_function(std::shared_ptr<ngraph::Function> f)
{
    // get parameters
    std::vector<std::shared_ptr<ngraph::op::Parameter>> back_parameters = f->get_parameters();

    ngraph::NodeVector adjoints;
    ngraph::NodeVector outputs;
    for (auto Y : f->get_results())
    {
        // Get the output
        // Create the Adjoint
        auto C = std::make_shared<ngraph::op::Parameter>(Y->get_element_type(), Y->get_shape());
        outputs.push_back(Y);
        adjoints.push_back(C);
    }

    ngraph::autodiff::Adjoints adjoint{outputs, adjoints};

    // Perform autodiff
    std::vector<std::shared_ptr<Node>> dYdXs(back_parameters.size());
    transform(back_parameters.begin(),
              back_parameters.end(),
              dYdXs.begin(),
              [&adjoint](const std::shared_ptr<Node>& X) { return adjoint.backprop_node(X); });

    // create the backward function
    std::vector<std::shared_ptr<ngraph::op::Parameter>> param_adjoints;
    for (auto n : adjoints)
        param_adjoints.push_back(std::dynamic_pointer_cast<ngraph::op::Parameter>(n));
    back_parameters.insert(back_parameters.begin(), param_adjoints.begin(), param_adjoints.end());

    return {std::make_shared<ngraph::Function>(dYdXs, back_parameters), adjoints};
}

void optimize_graph(std::shared_ptr<ngraph::Function>& f, std::shared_ptr<ngraph::Function> bf)
{
    // start by removing excess reshapes
    NodeVector nv_cwi;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
    pass_manager.register_pass<pass::VisualizeTree>("before.fprop_cache.pdf");

    pass_manager.run_passes(f);
    pass_manager.run_passes(bf);
    if (nv_cwi.size() > 0)
    {
        NodeVector new_outputs;
        for (auto r : f->get_results())
        {
            new_outputs.push_back(r->get_argument(0));
        }

        new_outputs.insert(new_outputs.end(), nv_cwi.begin(), nv_cwi.end());
        f = std::make_shared<ngraph::Function>(new_outputs, f->get_parameters());
    }

    ngraph::NodeVector dYdXs;
    for (size_t i = 0; i < bf->get_output_size(); ++i)
    {
        dYdXs.push_back(bf->get_output_op(i)->get_argument(0));
    }

    ngraph::NodeVector combined_outputs;
    for (auto r : f->get_results())
    {
        combined_outputs.push_back(r->get_argument(0));
    }

    combined_outputs.insert(combined_outputs.end(), dYdXs.begin(), dYdXs.end());

    std::vector<std::shared_ptr<ngraph::op::Parameter>> combined_parameters = f->get_parameters();
    std::vector<std::shared_ptr<ngraph::op::Parameter>> back_parameters = bf->get_parameters();

    combined_parameters.insert(
        combined_parameters.end(), back_parameters.begin(), back_parameters.end());
    auto combinedf = std::make_shared<ngraph::Function>(combined_outputs, combined_parameters);
    // rerun Reshape elimination to help simplify the graph again, run CPUFusion
    // this replaces nodes in both f and bf due to shared-ptr - ness
    ngraph::pass::Manager pass_manager_comb;
    pass_manager_comb.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager_comb.register_pass<ngraph::runtime::cpu::pass::CPUFusion>();
    pass_manager_comb.run_passes(combinedf);
}

TEST(cpu_fusion, maxpool_with_indices_in_mxnet)
{
    auto f = make_forward_function();
    auto bfa = make_backward_function(f);
    auto maybe_bf = bfa.first;
    auto adjoints = bfa.second;
    optimize_graph(f, maybe_bf);
    auto fprop_cache = ngraph::cache_fprop(f, maybe_bf);

    auto mpwi_bprop = fprop_cache.bprop->get_results().at(0)->get_argument(0);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Parameter>(mpwi_bprop->get_argument(0)));
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Parameter>(mpwi_bprop->get_argument(2)));
}

TEST(cpu_fusion, batch_norm_folding)
{
    Shape shape_input{1, 8, 3, 3};
    Shape shape_weights{2, 8, 1, 1};
    Shape shape_norm{2};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);
        double eps = 0.001;
        auto gamma = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto beta = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto mean = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto var = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto conv = std::make_shared<op::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto bn = std::make_shared<op::BatchNorm>(eps, gamma, beta, conv, mean, var);
        auto f = make_shared<Function>(NodeVector{bn},
                                       op::ParameterVector{input, weights, gamma, beta, mean, var});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {1.25f,
         2.25f,
         5.25f,
         6.25f,
         -1.25f,
         -1.25f,
         3.25f,
         -4.25f,
         7.25f,
         8.25f,
         -1.25f,
         0.f,
         0.f,
         0.f,
         0.f,
         -2.f},
        {-0.9384f, 0.01875f},
        {11.0f, 1.3f},
        {0.12f, 0.31f},
        {0.01f, 0.11f},
    };

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

TEST(cpu_fusion, affine_folding)
{
    Shape shape_input{1, 8, 3, 3};
    Shape shape_weights{2, 8, 1, 1};
    Shape shape_norm{2};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::Parameter>(element::f32, shape_weights);

        auto a = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto b = std::make_shared<op::Parameter>(element::f32, shape_norm);
        auto conv = std::make_shared<op::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto out = std::make_shared<op::Add>(
            std::make_shared<op::Multiply>(
                conv, std::make_shared<op::Broadcast>(a, conv->get_shape(), AxisSet{0, 2, 3})),
            std::make_shared<op::Broadcast>(b, conv->get_shape(), AxisSet{0, 2, 3}));
        auto f = make_shared<Function>(NodeVector{out}, op::ParameterVector{input, weights, a, b});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {1.25f,
         2.25f,
         5.25f,
         6.25f,
         -1.25f,
         -1.25f,
         3.25f,
         -4.25f,
         7.25f,
         8.25f,
         -1.25f,
         0.f,
         0.f,
         0.f,
         0.f,
         -2.f},
        {-0.9384f, 0.01875f},
        {11.0f, 1.3f},
    };

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

TEST(cpu_fusion, group_convolution_fusion)
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

    auto f = make_shared<Function>(NodeVector{concat}, op::ParameterVector{A, B});
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before_group.pdf");
    pass_manager.register_pass<runtime::cpu::pass::CPUBatchFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("after_group.pdf");
    pass_manager.run_passes(f);
    auto gc =
        std::dynamic_pointer_cast<op::GroupConvolution>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(gc);
}

TEST(cpu_fusion, group_convolution)
{
    auto backend = runtime::Backend::create("CPU");
    test::Uniform<float> rng(2.0f, 10.0f);

    const size_t GROUPS = 2;
    Shape shape_a{1, 32, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto group_conv = make_shared<op::GroupConvolution>(A,
                                                        B,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        GROUPS,
                                                        shape_r);

    Shape shape_c{1, 16, 2, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_d{1, 16, 1, 1};
    auto D = make_shared<op::Parameter>(element::f32, shape_d);
    auto conv_lower = make_shared<op::Convolution>(C,
                                                   D,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});

    auto E = make_shared<op::Parameter>(element::f32, shape_c);
    auto F = make_shared<op::Parameter>(element::f32, shape_d);
    auto conv_upper = make_shared<op::Convolution>(E,
                                                   F,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});

    auto f = make_shared<Function>(NodeVector{group_conv, conv_lower, conv_upper},
                                   op::ParameterVector{A, B, C, D, E, F});

    auto a_ = rng.initialize(backend->create_tensor(element::f32, shape_a));
    auto b_ = rng.initialize(backend->create_tensor(element::f32, shape_b));

    vector<float> rv(shape_size(shape_r), 0);
    auto group_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensorView>(
        backend->create_tensor(element::f32, shape_r, rv.data()));

    auto av = read_vector<float>(a_);
    auto bv = read_vector<float>(b_);
    auto c_ = backend->create_tensor(element::f32, shape_c, av.data()); // lower data
    auto d_ = backend->create_tensor(element::f32, shape_d, bv.data()); // upper data

    auto e_ =
        backend->create_tensor(element::f32, shape_c, av.data() + av.size() / 2); // lower weights
    auto f_ =
        backend->create_tensor(element::f32, shape_d, bv.data() + bv.size() / 2); // upper weights

    Shape shape_ur{1, 1, 2, 2};
    // allocate a contigious storage for both lower and upper halves.
    vector<float> erv(shape_size(shape_r), 0);
    auto lower_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensorView>(
        backend->create_tensor(element::f32, shape_ur, erv.data()));
    auto upper_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensorView>(
        backend->create_tensor(element::f32, shape_ur, erv.data() + erv.size() / 2));
    backend->call_with_validate(
        f, {group_result, lower_result, upper_result}, {a_, b_, c_, d_, e_, f_});
    ASSERT_EQ(rv, erv);
}

//TODO(Pruthvi) enable this test after MKLDNN RNN bug is fixed
#if 0
TEST(cpu_fusion, rnn_fprop_1_lstm_cell)
{
    auto src_layer = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto src_iter = make_shared<op::Parameter>(element::f32, Shape{20, 100});
    auto weights_layer = make_shared<op::Parameter>(element::f32, Shape{400, 100});
    auto weights_iter = make_shared<op::Parameter>(element::f32, Shape{400, 100});
    auto biases = make_shared<op::Parameter>(element::f32, Shape{400});
    const int number_of_timesteps = 1;
    const int number_of_gates_per_cell = 4;
    const int src_seq_length = 1;
    const int src_layer_feature_size = 100;
    const int feature_size = 100;
    const int num_rnn_cell_states = 2;
    const int rnn_direction = 1;
    const int num_of_rnn_fused_layer = 1;
    auto rnn_node = make_shared<op::Rnn>(src_layer,
                                         src_iter,
                                         weights_layer,
                                         weights_iter,
                                         biases,
                                         number_of_timesteps,
                                         number_of_gates_per_cell,
                                         src_seq_length,
                                         src_layer_feature_size,
                                         feature_size,
                                         num_rnn_cell_states,
                                         rnn_direction,
                                         num_of_rnn_fused_layer);
    auto rnn_ht_output = make_shared<op::GetOutputElement>(rnn_node, 0);
    auto rnn_ct_output = make_shared<op::GetOutputElement>(rnn_node, 1);

    auto func = make_shared<Function>(
        NodeVector{rnn_ht_output, rnn_ct_output},
        op::ParameterVector{src_layer, src_iter, weights_layer, weights_iter, biases});
    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::TensorView> src_layer_t =
        backend->create_tensor(element::f32, src_layer->get_shape());
    shared_ptr<runtime::TensorView> src_iter_t =
        backend->create_tensor(element::f32, src_iter->get_shape());
    shared_ptr<runtime::TensorView> weights_layer_t =
        backend->create_tensor(element::f32, weights_layer->get_shape());
    shared_ptr<runtime::TensorView> weights_iter_t =
        backend->create_tensor(element::f32, weights_iter->get_shape());
    shared_ptr<runtime::TensorView> biases_t =
        backend->create_tensor(element::f32, biases->get_shape());
    shared_ptr<runtime::TensorView> result_ht = backend->create_tensor(element::f32, {10, 100});
    shared_ptr<runtime::TensorView> result_ct =
        backend->create_tensor(element::f32, Shape{20, 100});

    copy_data(src_layer_t, vector<float>(1000, 1));
    copy_data(src_iter_t, vector<float>(2000, 1));
    copy_data(weights_layer_t, vector<float>(400 * 100, 1));
    copy_data(weights_iter_t, vector<float>(400 * 100, 1));
    copy_data(biases_t, vector<float>(400, 1));

    backend->call_with_validate(
        func,
        {result_ht, result_ct},
        {src_layer_t, src_iter_t, weights_layer_t, weights_iter_t, biases_t});
    vector<float> expected_ht(10 * 100, 0.964028f);
    vector<float> expected_ct;
    for (size_t i = 0; i < 20 * 100; i++)
    {
        if (i < 1000)
        {
            expected_ct.push_back(0.964028f);
        }
        else
        {
            expected_ct.push_back(2.0f);
        }
    }

    EXPECT_TRUE(test::all_close(expected_ht, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(expected_ct, read_vector<float>(result_ct)));
}
#endif

TEST(cpu_fusion, fuse_lstm_cells)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::ConcatInputs>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    auto lstm_ops = get_ops_of_type<op::Lstm>(func);
    EXPECT_EQ(lstm_ops.size(), 6);
}

TEST(cpu_fusion, fuse_2_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
        EXPECT_EQ(node->get_num_cell_states(), node->get_argument(1)->get_arguments().size());
    }
}

TEST(cpu_fusion, fuse_1_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/1rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), 1);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
        EXPECT_EQ(node->get_num_cell_states(), node->get_argument(1)->get_arguments().size());
    }
}

static std::shared_ptr<Function> make_function(const std::string& file_name)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, file_name);
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    return func;
}

TEST(cpu_fusion, rnn_fusion_inter_vs_cpu_1lstm_cell)
{
    const std::string file_name("mxnet/1_lstm_cell_forward.json");
    auto cpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, rnn_fusion_inter_vs_cpu_1rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/1rnn_layer_3lstm_cell.json");
    auto cpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, rnn_fusion_inter_vs_cpu_2rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/2rnn_layer_3lstm_cell.json");
    auto cpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, loop_kernel_fusion_multiple_groups_pruned)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        Shape shape{};
        auto a = make_shared<op::Parameter>(element::f32, shape);
        auto b = make_shared<op::Parameter>(element::f32, shape);
        auto c = make_shared<op::Parameter>(element::f32, shape);
        auto add_ab = a + b;
        auto add_abs = std::make_shared<op::Abs>(add_ab);
        auto abs_neg = std::make_shared<op::Negative>(add_abs);
        auto sub_c_neg = c - abs_neg;

        auto d = make_shared<op::Parameter>(element::f32, shape);
        auto d_abs = std::make_shared<op::Abs>(d);
        auto add_d = d_abs + add_ab;
        auto neg_d = std::make_shared<op::Negative>(add_d);

        auto mul_cd = neg_d * sub_c_neg;
        auto f =
            std::make_shared<Function>(ngraph::NodeVector{mul_cd}, op::ParameterVector{a, b, c, d});

        return f;
    };

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPULoopKernelFusion>(3);
    auto cpu_f = make_function();
    auto int_f = make_function();
    pass_manager.run_passes(cpu_f);
    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;

    size_t lkn = count_ops_of_type<runtime::cpu::op::LoopKernel>(cpu_f);
    ASSERT_GT(lkn, 0);

    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, loop_kernel_fusion_bounded_relu)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        Shape shape{};
        auto a = make_shared<op::Parameter>(element::f32, shape);
        auto relu = make_shared<op::Relu>(a);
        auto upper_bound =
            op::Constant::create<float>(element::f32, shape, std::vector<float>{6.0f});
        auto minn = make_shared<op::Minimum>(relu, upper_bound);
        auto absn = make_shared<op::Abs>(minn);
        auto negn = std::make_shared<op::Negative>(absn);

        auto f = std::make_shared<Function>(ngraph::NodeVector{negn}, op::ParameterVector{a});

        return f;
    };

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before_relu_fusion.pdf");
    pass_manager.register_pass<runtime::cpu::pass::CPULoopKernelFusion>(3);
    pass_manager.register_pass<pass::VisualizeTree>("after_relu_fusion.pdf");
    auto cpu_f = make_function();
    auto int_f = make_function();
    pass_manager.run_passes(cpu_f);
    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;

    size_t lkn = count_ops_of_type<runtime::cpu::op::LoopKernel>(cpu_f);
    ASSERT_GT(lkn, 0);

    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, loop_kernel_fusion_multiple_groups)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        Shape shape{};
        auto a = make_shared<op::Parameter>(element::f32, shape);
        auto b = make_shared<op::Parameter>(element::f32, shape);
        auto c = make_shared<op::Parameter>(element::f32, shape);
        auto add_ab = a + b;
        auto add_abs = std::make_shared<op::Abs>(add_ab);
        auto abs_neg = std::make_shared<op::Negative>(add_abs);
        auto sub_c_neg = c - abs_neg;

        auto d = make_shared<op::Parameter>(element::f32, shape);
        auto d_abs = std::make_shared<op::Abs>(d);
        auto add_d = d_abs + add_ab;
        auto neg_d = std::make_shared<op::Negative>(add_d);

        auto mul_cd = neg_d * sub_c_neg;
        auto f =
            std::make_shared<Function>(ngraph::NodeVector{mul_cd}, op::ParameterVector{a, b, c, d});

        return f;
    };

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPULoopKernelFusion>(2);
    auto cpu_f = make_function();
    auto int_f = make_function();
    pass_manager.run_passes(cpu_f);
    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;

    size_t lkn = count_ops_of_type<runtime::cpu::op::LoopKernel>(cpu_f);
    ASSERT_GT(lkn, 0);

    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, loop_kernel_fusion_one_group)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        Shape shape{};
        auto a = make_shared<op::Parameter>(element::f32, shape);
        auto b = make_shared<op::Parameter>(element::f32, shape);
        auto c = make_shared<op::Parameter>(element::f32, shape);
        auto add_ab = a + b;
        auto add_abs = std::make_shared<op::Abs>(add_ab);
        auto abs_neg = std::make_shared<op::Negative>(add_abs);
        auto sub_c_neg = c - abs_neg;
        auto d = make_shared<op::Parameter>(element::f32, shape);
        auto add_d = sub_c_neg + d;
        auto abs_add_d = std::make_shared<op::Abs>(add_d);
        auto e = make_shared<op::Parameter>(element::f32, shape);
        auto add_e = e + abs_add_d;
        auto neg_e = std::make_shared<op::Negative>(add_e);

        auto f = std::make_shared<Function>(ngraph::NodeVector{neg_e},
                                            op::ParameterVector{a, b, c, d, e});

        return f;

    };

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPULoopKernelFusion>(2);
    auto cpu_f = make_function();
    auto int_f = make_function();
    pass_manager.run_passes(cpu_f);
    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;

    size_t lkn = count_ops_of_type<runtime::cpu::op::LoopKernel>(cpu_f);
    ASSERT_GT(lkn, 0);

    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, sigmoid_multiply_fusion)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/3_lstm_cell_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::SigmoidMultiply>(func);
    ASSERT_EQ(ccg, 18);
}

void sigmoid_multiply_fusion_forward_compute(shared_ptr<runtime::Backend>& backend,
                                             const op::ParameterVector& input_params,
                                             const vector<vector<float>>& input_data,
                                             const vector<Shape>& input_shapes,
                                             const Shape& result_shape,
                                             shared_ptr<Node> input_0_node,
                                             shared_ptr<Node> input_1_node,
                                             const vector<float>& expected)
{
    shared_ptr<runtime::TensorView> result_tensor =
        backend->create_tensor(element::f32, result_shape);

    vector<shared_ptr<runtime::TensorView>> input_tensors;
    for (int i = 0; i < input_params.size(); ++i)
    {
        input_tensors.push_back(backend->create_tensor(element::f32, input_shapes[i]));
        copy_data(input_tensors[i], input_data[i]);
    }

    auto mul_node = input_0_node * input_1_node;
    auto func = make_shared<Function>(mul_node, input_params);
    backend->call_with_validate(func, {result_tensor}, input_tensors);
    EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor), expected));
}

TEST(cpu_fusion, sigmoid_multiply_fusion_forward)
{
    auto backend = runtime::Backend::create("CPU");

    Shape data_shape{1, 1, 2, 2};
    Shape const_shape{1};

    vector<float> input_0_data{1.f, 2.f, 3.f, 4.f};
    vector<float> input_1_data{1.2f, 2.3f, 3.5f, 4.7f};
    vector<float> const_data{1.2f};
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_2_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Add>(input_1_param, input_2_param);
        vector<float> expected{1.60833f, 3.78743f, 6.19173f, 8.54352f};
        op::ParameterVector input_params{input_0_param, input_1_param, input_2_param};
        vector<vector<float>> input_data{input_0_data, input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        auto sigmoid_1 = make_shared<op::Sigmoid>(input_0_param);
        vector<float> expected{0.87727f, 1.05696f, 1.14309f, 1.17842f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        vector<float> expected{0.87727f, 1.05696f, 1.14309f, 1.17842f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Sigmoid>(input_1_param);
        vector<float> expected{0.561837f, 0.800536f, 0.924652f, 0.973163f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Tanh>(input_1_param);
        vector<float> expected{0.60945f, 0.863266f, 0.950838f, 0.981851f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::Sigmoid>(input_1_param);
        vector<float> expected{0.585304f, 0.876182f, 0.965887f, 0.990322f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::Tanh>(input_1_param);
        vector<float> expected{0.634907f, 0.94484f, 0.993242f, 0.999164f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend,
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
}

void sigmoid_multiply_fusion_backward_compute(shared_ptr<runtime::Backend>& backend,
                                              const op::ParameterVector& input_params,
                                              const vector<vector<float>>& input_data,
                                              const vector<Shape>& input_shapes,
                                              const vector<float> delta_data,
                                              const Shape& delta_shape,
                                              const Shape& d_input_0_shape,
                                              const Shape& d_input_1_shape,
                                              shared_ptr<Node> input_0_node,
                                              shared_ptr<Node> input_1_node,
                                              shared_ptr<Node> input_0_adjoint,
                                              shared_ptr<Node> input_1_adjoint,
                                              const vector<float>& expected_0,
                                              const vector<float>& expected_1)
{
    vector<shared_ptr<runtime::TensorView>> input_tensors;
    for (int i = 0; i < input_params.size(); ++i)
    {
        input_tensors.push_back(backend->create_tensor(element::f32, input_shapes[i]));
        copy_data(input_tensors[i], input_data[i]);
    }

    auto delta_param = make_shared<op::Parameter>(element::f32, delta_shape);
    shared_ptr<runtime::TensorView> delta_tensor =
        backend->create_tensor(element::f32, delta_shape);
    copy_data(delta_tensor, delta_data);

    op::ParameterVector back_params(input_params);
    back_params.push_back(delta_param);
    input_tensors.push_back(delta_tensor);

    shared_ptr<runtime::TensorView> d_input_0_tensor =
        backend->create_tensor(element::f32, d_input_0_shape);
    shared_ptr<runtime::TensorView> d_input_1_tensor =
        backend->create_tensor(element::f32, d_input_1_shape);

    using FunctionType = op::SigmoidMultiply::FunctionType;
    auto input_0_type = op::SigmoidMultiply::identify_node_type(input_0_node);
    auto input_1_type = op::SigmoidMultiply::identify_node_type(input_1_node);
    // for Identity functions, we use the node itself, otherwise use its input
    // where we will apply the function of input node
    auto input_0_alt =
        (input_0_type == FunctionType::Identity) ? input_0_node : input_0_node->get_argument(0);
    auto input_1_alt =
        (input_1_type == FunctionType::Identity) ? input_1_node : input_1_node->get_argument(0);
    auto sigmoid_mul =
        make_shared<op::SigmoidMultiply>(input_0_alt, input_1_alt, input_0_type, input_1_type);

    ngraph::autodiff::Adjoints adjoints(NodeVector{sigmoid_mul}, NodeVector{delta_param});
    auto d_input_0 = adjoints.backprop_node(input_0_adjoint);
    auto d_input_1 = adjoints.backprop_node(input_1_adjoint);
    auto df = make_shared<Function>(NodeVector{d_input_0, d_input_1}, back_params);
    backend->call_with_validate(df, {d_input_0_tensor, d_input_1_tensor}, input_tensors);
    EXPECT_TRUE(test::all_close(read_vector<float>(d_input_0_tensor), expected_0));
    EXPECT_TRUE(test::all_close(read_vector<float>(d_input_1_tensor), expected_1));
}

TEST(cpu_fusion, sigmoid_multiply_fusion_backward)
{
    auto backend = runtime::Backend::create("CPU");

    Shape data_shape{1, 1, 2, 2};
    Shape const_shape{1};

    vector<float> input_0_data{1.f, 2.f, 3.f, 4.f};
    vector<float> input_1_data{1.2f, 2.2f, 3.2f, 4.2f};
    vector<float> const_data{1.2f};
    vector<float> delta_data(shape_size(data_shape), 20.0f);

    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_2_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Add>(input_1_param, input_2_param);
        vector<float> expected_0{8.65093f, 8.81946f, 5.60191f, 2.89668f};
        vector<float> expected_1{14.6212f, 17.6159f, 19.0515f, 19.6403f};
        op::ParameterVector input_params{input_0_param, input_1_param, input_2_param};
        vector<vector<float>> input_data{input_0_data, input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 sigmoid_1,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        auto sigmoid_1 = make_shared<op::Tanh>(input_0_param);
        vector<float> expected_0{15.2319f, 19.2806f, 19.9011f, 19.9866f};
        vector<float> expected_1{10.0794f, 1.69562f, 0.236785f, 0.0321828f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 sigmoid_0,
                                                 input_0_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        vector<float> expected_0{10.0794f, 1.69562f, 0.236785f, 0.0321828f};
        vector<float> expected_1{15.2319f, 19.2806f, 19.9011f, 19.9866f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 sigmoid_1,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Sigmoid>(input_1_param);
        vector<float> expected_0{3.02202f, 1.89041f, 0.868146f, 0.348035f};
        vector<float> expected_1{2.60102f, 1.58192f, 0.716941f, 0.285879f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::Tanh>(input_1_param);
        vector<float> expected_0{3.27813f, 2.04894f, 0.900536f, 0.353095f};
        vector<float> expected_1{4.45975f, 0.84425f, 0.126201f, 0.0176579f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::Sigmoid>(input_1_param);
        vector<float> expected_0{6.45521f, 1.27207f, 0.189593f, 0.0264228f};
        vector<float> expected_1{2.70967f, 1.7314f, 0.748913f, 0.29092f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::Tanh>(input_1_param);
        vector<float> expected_0{7.00227f, 1.37874f, 0.196666f, 0.026807f};
        vector<float> expected_1{4.64603f, 0.924027f, 0.131829f, 0.0179692f};
        op::ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend,
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
}

TEST(cpu_fusion, fuse_batch_dot)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUBatchFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/batch_dot_3.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::BatchDot>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(cpu_fusion, fuse_batch_dot_forward)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUBatchFusion>();

    const std::string file_name("mxnet/batch_dot_3.json");
    auto cpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    pass_manager.run_passes(cpu_f);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    for (size_t i = 0; i < int_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_fusion, fuse_rnn_across_layer)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::cpu::pass::MultiLayerRNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_1timestep.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ref_rnn_count = 1;
    auto rnn_count = count_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(ref_rnn_count, rnn_count);
}

TEST(cpu_fusion, fuse_rnn_across_2layer_1timestep)
{
    const std::string file_name("mxnet/2rnn_layer_1timestep.json");
    auto cpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    // TODO (pruthvi): Enable this after fixing failing
    // mxnet rnn unit tests
    // EXPECT_EQ(1, count_ops_of_type<op::Rnn>(cpu_f));
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(1), int_results.at(1), 1.0e-4f, 1.0e-4f));
    }
}

static void check_bounded_relu(Shape param_shape, float constant_val)
{
    auto make_function = [](Shape input_shape, float alpha_val) {
        auto relu_input = std::make_shared<op::Parameter>(element::f32, input_shape);
        auto relu = std::make_shared<op::Relu>(relu_input);
        auto alpha = op::Constant::create<float>(
            element::f32, input_shape, std::vector<float>(1.0f, alpha_val));
        auto min = std::make_shared<op::Minimum>(relu, alpha);
        auto f = make_shared<Function>(NodeVector{min}, op::ParameterVector{relu_input});
        return f;
    };

    auto cpu_f = make_function(param_shape, constant_val);
    auto int_f = make_function(param_shape, constant_val);
    test::Uniform<float> rng(-10.0f, 10.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    EXPECT_EQ(1, count_ops_of_type<op::BoundedRelu>(cpu_f));
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0), 1.0e-4f, 1.0e-4f));
}

TEST(cpu_fusion, fuse_bounded_relu_inter_vs_cpu)
{
    check_bounded_relu(Shape{4, 3, 2, 2}, 6.0f);
    check_bounded_relu(Shape{4, 3}, 4.0f);
    check_bounded_relu(Shape{4, 3, 2}, 2.0f);
}

TEST(cpu_fusion, dot_batch_forward)
{
    const Shape shape_a{2, 3, 2};
    const Shape shape_b{2, 3};

    auto generate_func = [&shape_a, &shape_b]() -> shared_ptr<Function> {
        auto a = make_shared<op::Parameter>(element::f32, shape_a);
        auto b = make_shared<op::Parameter>(element::f32, shape_b);
        auto dot = make_shared<op::Dot>(a, b);
        return make_shared<Function>(dot, op::ParameterVector{a, b});
    };
    shared_ptr<Function> cpu_func = generate_func();
    shared_ptr<Function> int_func = generate_func();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : int_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_func, args, "INTERPRETER");
    auto cpu_results = execute(cpu_func, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}
