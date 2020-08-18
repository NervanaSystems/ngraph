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
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_no_bias_no_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell =
        make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");
    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_zero_bias_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::v0::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::v0::LSTMCell>(
        X, H_t, C_t, W, R, B, P, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B(gates_count * hidden_size, 0.f);
    // P
    vector<float> in_P(3 * hidden_size, 0.f);

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_zero_bias_peepholes_constant)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Constant>(
        element::f32, Shape{gates_count * hidden_size}, std::vector<float>{0.f});
    const auto P = make_shared<op::v0::Constant>(
        element::f32, Shape{3 * hidden_size}, std::vector<float>{0.f});

    const auto lstm_cell = make_shared<op::v0::LSTMCell>(
        X, H_t, C_t, W, R, B, P, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_fixed_no_bias_no_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell =
        make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X(batch_size * input_size, 0.5f);
    // W
    vector<float> in_W(gates_count * hidden_size * input_size, 0.25f);
    // R
    vector<float> in_R(gates_count * hidden_size * hidden_size, 0.25f);
    // Ht
    vector<float> in_Ht(batch_size * hidden_size, 0.75f);
    // Ct
    vector<float> in_Ct(batch_size * hidden_size, 0.75f);

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.56633735f, 0.56633735f, 0.56633735f, 0.56633735f, 0.56633735f, 0.56633735f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.0664454f, 1.0664454f, 1.0664454f, 1.0664454f, 1.0664454f, 1.0664454f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::v0::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::v0::LSTMCell>(
        X, H_t, C_t, W, R, B, P, hidden_size, op::LSTMWeightsFormat::IOFC);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9218244f, 0.78787273f, 0.8754273f, 0.7361462f, 0.70927656f, 0.83522964f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.7094649f, 1.1259761f, 1.444019f, 1.086587f, 0.9762144f, 1.3066899f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes_clip_input_forget)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::v0::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::v0::LSTMCell>(X,
                                                         H_t,
                                                         C_t,
                                                         W,
                                                         R,
                                                         B,
                                                         P,
                                                         hidden_size,
                                                         op::LSTMWeightsFormat::IOFC,
                                                         vector<string>{"sigmoid", "tanh", "tanh"},
                                                         vector<float>{},
                                                         vector<float>{},
                                                         clip_threshold,
                                                         input_forget);
    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.71485436f, 0.71844107f, 0.72704613f, 0.6235602f, 0.68306124f, 0.6978715f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_activaction_functions)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;
    vector<string> activations{"sigmoid", "tanh", "hardsigmoid"};
    vector<float> activation_alpha{0.f, 0.f, 1.8345f};
    vector<float> activation_beta{0.f, 0.f, 3.05f};

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto P = make_shared<op::v0::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::v0::LSTMCell>(X,
                                                         H_t,
                                                         C_t,
                                                         W,
                                                         R,
                                                         B,
                                                         P,
                                                         hidden_size,
                                                         op::LSTMWeightsFormat::IOFC,
                                                         activations,
                                                         activation_alpha,
                                                         activation_beta,
                                                         clip_threshold,
                                                         input_forget);
    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.96834344f, 0.9695254f, 0.97068775f, 0.9077866f, 0.94161016f, 0.96599925f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}
