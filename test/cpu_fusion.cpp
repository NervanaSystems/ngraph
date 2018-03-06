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
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
//
#include "ngraph/file_util.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/ops/conv_bias.hpp"
#include "ngraph/runtime/cpu/ops/matmul_bias.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/matcher.hpp"
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

    auto cg =
        make_shared<op::MatmulBias>(W, x, broadcast, W->get_shape(), x->get_shape(), false, false);
}

TEST(cpu_fusion, gemm_cpu)
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
    auto cg =
        make_shared<op::MatmulBias>(A, B, broadcast, A->get_shape(), B->get_shape(), true, true);

    auto f = make_shared<Function>(cg, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shapeA);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shapeB);
    shared_ptr<runtime::TensorView> result =
        backend->make_primary_tensor_view(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    cf->call({a, b}, {result});
    vector<float> expected{10, 28, 37, 109};
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

TEST(cpu_fusion, gemm_mlp)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/mnist_mlp_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(ccg, 3);
}

//TODO: Move this test to backend_test.in.cpp once we have the INTERPRETER
//      implementation for batchnorm
TEST(cpu_fusion, batchnorm_fprop_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var);

    auto f = make_shared<Function>(bn, op::ParameterVector{mean, var, input, gamma, beta});
    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto _input = backend->make_primary_tensor_view(element::f32, Shape{1, 2, 2, 2});

    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto _mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.60291237f, 0.59972727f});
    auto _var = backend->make_primary_tensor_view(element::f32, var_shape);
    copy_data(_var, vector<float>{0.00472505f, 0.03617825f});
    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    cf->call({_mean, _var, _input, _gamma, _beta}, {result});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

TEST(cpu_fusion, batchnorm_fprop_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var);

    auto f = make_shared<Function>(bn, op::ParameterVector{mean, var, input, gamma, beta});
    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto _input = backend->make_primary_tensor_view(element::f32, Shape{2, 2, 2, 1});
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto _mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.60291237f, 0.59972727f});
    auto _var = backend->make_primary_tensor_view(element::f32, var_shape);
    copy_data(_var, vector<float>{0.00472505f, 0.03617825f});
    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    vector<float> expected_result{
        -0.714987f, 1.48389f, 0.015746f, -0.284436f, -2.36912f, 0.56806f, -0.840903f, 1.51463f};
    cf->call({_mean, _var, _input, _gamma, _beta}, {result});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
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

class UnhandledOp : public ngraph::op::Abs
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Abs(arg)
    {
    }
};

TEST(cpu_fusion, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();
    auto external = manager->compile(f);
    ASSERT_THROW(backend->make_call_frame(external), ngraph_error);
}

TEST(cpu_fusion, bn_bprop_n4c3h2w2)
{
    auto input_shape = Shape{4, 3, 2, 2};
    auto shape_mean = Shape{3};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{3};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{3};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{3};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{3};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{4, 3, 2, 2};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var);

    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();

    auto _input = backend->make_primary_tensor_view(element::f32, input_shape);
    vector<float> dataInput{
        10.76331902f, 11.51178265f, 10.31018162f, 12.2993021f,  14.17626667f, 14.63498497f,
        13.63494492f, 13.84248161f, 11.34602547f, 13.22014618f, 10.46686649f, 10.39842987f,
        12.94806862f, 11.71670246f, 14.94438076f, 13.13236618f, 13.40889645f, 12.76128387f,
        11.34430027f, 11.86629677f, 11.11464024f, 10.93221283f, 11.95324039f, 10.96581173f,
        13.05455494f, 14.41404247f, 13.11169434f, 11.26559448f, 10.89965153f, 14.08202171f,
        11.12685776f, 12.58428574f, 12.59247875f, 13.00187492f, 12.66310215f, 10.06655025f,
        12.62048626f, 14.47942352f, 13.84950638f, 10.61425877f, 11.47936344f, 13.06011772f,
        13.63069057f, 12.31748772f, 13.84555244f, 10.95815468f, 12.78933334f, 12.75389099f};
    copy_data(_input, dataInput);
    auto _mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    copy_data(_mean, vector<float>{12.56472874f, 12.80312157f, 11.81676865f});
    auto _var = backend->make_primary_tensor_view(element::f32, var_shape);
    copy_data(_var, vector<float>{1.94557643f, 1.32772446f, 1.28163588f});

    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{2.0f, 2.0f, 2.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{1.0f, 1.0f, 1.0f});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    shared_ptr<runtime::TensorView> _delta =
        backend->make_primary_tensor_view(element::f32, shape_r);
    vector<float> deltaData(shape_size(shape_r), 20.0f);
    copy_data(_delta, deltaData);

    auto f = make_shared<Function>(bn, op::ParameterVector{mean, var, input, gamma, beta});

    auto C = std::make_shared<op::Parameter>(element::f32, shape_r);
    auto dinput = bn->backprop_node(input, C);
    auto dgamma = bn->backprop_node(gamma, C);
    auto dbeta = bn->backprop_node(beta, C);

    auto df = make_shared<Function>(NodeVector{dinput, dgamma, dbeta},
                                    op::ParameterVector{mean, var, input, gamma, beta, C});

    //roundtrip serialization
    string js = serialize(df, 4);
    istringstream in(js);
    df = deserialize(in);

    auto external = manager->compile(df);
    auto cf = backend->make_call_frame(external);

    shared_ptr<runtime::TensorView> _dinput =
        backend->make_primary_tensor_view(element::f32, shape_r);
    shared_ptr<runtime::TensorView> _dgamma =
        backend->make_primary_tensor_view(element::f32, gamma_shape);
    shared_ptr<runtime::TensorView> _dbeta =
        backend->make_primary_tensor_view(element::f32, beta_shape);

    cf->call({_mean, _var, _input, _gamma, _beta, _delta}, {_dinput, _dgamma, _dbeta});

    vector<float> expected_input{
        8.17051607e-06f,  4.77576657e-06f,  1.02257760e-05f,  1.20387525e-06f,  -1.73868522e-06f,
        3.84632768e-06f,  -1.07932050e-05f, -2.57458956e-06f, -2.22166714e-06f, -8.38779043e-06f,
        -2.48082982e-06f, 5.89238360e-06f,  -2.52895109e-07f, -8.68433445e-06f, -5.82726737e-06f,
        8.84659658e-06f,  3.03944108e-05f,  4.05480879e-05f,  1.84123158e-05f,  2.30061178e-05f,
        1.34087590e-05f,  -9.26072571e-07f, -3.22908454e-05f, -2.07365116e-05f, -4.21330941e-05f,
        2.83083100e-05f,  -3.71039101e-05f, -4.84390640e-06f, -2.93012376e-05f, 5.68858087e-06f,
        1.83181458e-05f,  -1.07494506e-05f, -2.32429103e-06f, 6.92914809e-06f,  -6.66512321e-06f,
        -7.00302840e-06f, -3.46675184e-06f, -4.36748381e-06f, 6.73822226e-07f,  -4.20158993e-06f,
        3.83005061e-06f,  5.85143729e-06f,  4.17875243e-06f,  -8.64167783e-06f, 1.00170803e-05f,
        -4.23939666e-06f, 4.80201680e-06f,  4.62702078e-06f};

    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dinput), expected_input, 1e-3f, 1e-4f));
    vector<float> expected_dgamma{7.06315041e-05f, -2.35289335e-04f, -5.06639481e-05f};
    ASSERT_TRUE(
        ngraph::test::all_close(read_vector<float>(_dgamma), expected_dgamma, 1e-2f, 1e-3f));
    vector<float> expected_dbeta{320.f, 320.f, 320.f};
    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dbeta), expected_dbeta, 1e-4f, 1e-8f));
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

TEST(cpu_fusion, conv_bias_fprop)
{
    const int n = 1;
    const int c = 1;
    const int filter = 1;
    const int kernel_size = 3;
    const int w = 3;
    const int h = w;

    auto data_shape = Shape{n, c, h, w};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto weights_shape = Shape{filter, c, kernel_size, kernel_size};
    auto weights = make_shared<op::Parameter>(element::f32, weights_shape);
    auto bias_shape = Shape{filter};
    auto bias = make_shared<op::Parameter>(element::f32, bias_shape);
    auto convolution= make_shared<op::Convolution>(data, weights);
    auto convolution_bias = make_shared<op::ConvolutionBias>(convolution, bias);
    auto result_shape = Shape{1,1,1,1};

    auto f = make_shared<Function>(convolution_bias, op::ParameterVector{data, weights, bias});
    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto _data = backend->make_primary_tensor_view(element::f32, data_shape);
    copy_data(_data, vector<float>{-0.67765152, 0.10073948, 0.57595438,
                                   -0.3469252, -0.22134334, -1.80471897,
                                   -0.80642909, 1.22033095, 2.23235631});
    auto _weights = backend->make_primary_tensor_view(element::f32, weights_shape);
    copy_data(_weights, vector<float>{1,1,1,
                                      1,1,1,
                                      1,1,1});
    auto _bias = backend->make_primary_tensor_view(element::f32, bias_shape);
    copy_data(_bias, vector<float>{1});
    auto result = backend->make_primary_tensor_view(element::f32, result_shape);

//    vector<float> expected_result{-0.71498716f,
//                                  1.48388731f,
//                                  -0.00196938f,
//                                  -0.76693159f,
//                                  -0.91316032f,
//                                  0.23943391f,
//                                  -0.84090298f,
//                                  1.51462936f};
    cf->call({_data, _weights, _bias}, {result});
    auto result_vec = read_vector<float>(result);
    for (size_t i = 0; i < result_vec.size(); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
    //EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

TEST(cpu_fusion, conv_bias_bprop)
{
    const int n = 1;
    const int c = 1;
    const int filter = 1;
    const int kernel_size = 3;
    const int w = 3;
    const int h = w;

    auto data_shape = Shape{n, c, h, w};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto weights_shape = Shape{filter, c, kernel_size, kernel_size};
    auto weights = make_shared<op::Parameter>(element::f32, weights_shape);
    auto bias_shape = Shape{filter};
    auto bias = make_shared<op::Parameter>(element::f32, bias_shape);
    auto convolution= make_shared<op::Convolution>(data, weights);
    auto convolution_bias = make_shared<op::ConvolutionBias>(convolution, bias);
    auto delta_shape = Shape{1,1,1,1};
    auto delta = std::make_shared<op::Parameter>(element::f32, delta_shape);

    auto f = make_shared<Function>(convolution_bias, op::ParameterVector{data, weights, bias});
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();

    auto d_data = convolution_bias->backprop_node(data, delta);
    auto d_weights = convolution_bias->backprop_node(weights, delta);
    auto d_bias = convolution_bias->backprop_node(bias, delta);

    auto df = make_shared<Function>(NodeVector{d_data, d_weights, d_bias},
                                    op::ParameterVector{data, weights, bias, delta});

    auto external = manager->compile(df);
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto _data = backend->make_primary_tensor_view(element::f32, data_shape);
    copy_data(_data, vector<float>{-0.67765152, 0.10073948, 0.57595438,
                                   -0.3469252, -0.22134334, -1.80471897,
                                   -0.80642909, 1.22033095, 2.23235631});
    auto _weights = backend->make_primary_tensor_view(element::f32, weights_shape);
    copy_data(_weights, vector<float>{1,1,1,
                                      1,1,1,
                                      1,1,1});
    auto _bias = backend->make_primary_tensor_view(element::f32, bias_shape);
    copy_data(_bias, vector<float>{1});

    auto _delta = backend->make_primary_tensor_view(element::f32, delta_shape);
    copy_data(_delta, vector<float>{1.27231});

    // results
    auto _d_data = backend->make_primary_tensor_view(element::f32, data_shape);
    copy_data(_d_data, vector<float>{0,0,0,
                                     0,0,0,
                                     0,0,0});

    auto _d_weights = backend->make_primary_tensor_view(element::f32, weights_shape);
    copy_data(_d_weights, vector<float>{0,0,0,
                                        0,0,0,
                                        0,0,0});

    auto _d_bias = backend->make_primary_tensor_view(element::f32, bias_shape);
    copy_data(_d_bias, vector<float>{0});

//    vector<float> expected_result{-0.71498716f,
//                                  1.48388731f,
//                                  -0.00196938f,
//                                  -0.76693159f,
//                                  -0.91316032f,
//                                  0.23943391f,
//                                  -0.84090298f,
//                                  1.51462936f};
    cf->call({_data, _weights, _bias, _delta}, {_d_data, _d_weights, _d_bias});
    auto result_vec = read_vector<float>(_d_data);
    for (size_t i = 0; i < result_vec.size(); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
    result_vec = read_vector<float>(_d_weights);
    for (size_t i = 0; i < result_vec.size(); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
    result_vec = read_vector<float>(_d_bias);
    for (size_t i = 0; i < result_vec.size(); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
    //EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}