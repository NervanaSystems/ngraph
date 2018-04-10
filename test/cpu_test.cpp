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
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

//TODO: Move this test to backend_test.in.cpp once we have the INTERPRETER
//      implementation for batchnorm
TEST(cpu_test, batchnorm_fprop_b1c2h2w2)
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
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   op::ParameterVector{input, gamma, beta});
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
    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->make_primary_tensor_view(element::f32, shape_r);
    auto result_mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    auto result_variance = backend->make_primary_tensor_view(element::f32, var_shape);

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    vector<float> expected_mean{0.602912f, 0.599727f};
    vector<float> expected_variance{0.00472505f, 0.0361782f};

    cf->call({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(test::all_close(expected_variance, read_vector<float>(result_variance)));
}

TEST(cpu_test, batchnorm_fprop_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   op::ParameterVector{input, gamma, beta});
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

    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->make_primary_tensor_view(element::f32, shape_r);
    auto result_mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    auto result_variance = backend->make_primary_tensor_view(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    cf->call({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(test::all_close(expected_variance, read_vector<float>(result_variance)));
}

class UnhandledOp : public ngraph::op::Abs
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Abs(arg)
    {
    }
};

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();
    auto external = manager->compile(f);
    ASSERT_THROW(backend->make_call_frame(external), ngraph_error);
}

TEST(cpu_test, bn_bprop_n4c3h2w2)
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
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);
    auto bn_dx = make_shared<op::GetOutputElement>(bn, 0);
    auto bn_dgamma = make_shared<op::GetOutputElement>(bn, 1);
    auto bn_dbeta = make_shared<op::GetOutputElement>(bn, 2);

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

    auto f = make_shared<Function>(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                   op::ParameterVector{mean, var, input, gamma, beta});

    auto C = std::make_shared<op::Parameter>(element::f32, shape_r);

    auto zero = ngraph::make_zero(bn_dgamma->get_element_type(), bn_dgamma->get_shape());
    ngraph::autodiff::Adjoints adjoints(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                        NodeVector{C, zero, zero});

    auto dinput = adjoints.backprop_node(input);
    auto dgamma = adjoints.backprop_node(gamma);
    auto dbeta = adjoints.backprop_node(beta);

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

    cf->call({_dinput, _dgamma, _dbeta}, {_mean, _var, _input, _gamma, _beta, _delta});

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

TEST(cpu_test, batchnorm_fprop_inference_b2c2h2w1)
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

    auto f = make_shared<Function>(bn, op::ParameterVector{input, gamma, beta, mean, var});
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

    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->make_primary_tensor_view(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->make_primary_tensor_view(element::f32, shape_r);
    auto result_mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    auto result_variance = backend->make_primary_tensor_view(element::f32, var_shape);
    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    cf->call({bn_output}, {_input, _gamma, _beta, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

TEST(cpu_test, batchnorm_fprop_globalstats_b2c2w2h1)
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
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var, true);

    auto f = make_shared<Function>(bn, op::ParameterVector{gamma, beta, input, mean, var});
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

    auto _gamma = backend->make_primary_tensor_view(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->make_primary_tensor_view(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->make_primary_tensor_view(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->make_primary_tensor_view(element::f32, shape_r);
    auto result_mean = backend->make_primary_tensor_view(element::f32, mean_shape);
    auto result_variance = backend->make_primary_tensor_view(element::f32, var_shape);
    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    cf->call({bn_output}, {_gamma, _beta, _input, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}
