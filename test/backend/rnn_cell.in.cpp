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

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_no_bias)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<op::v0::RNNCell>(X, H_t, W, R, hidden_size);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9408395f, 0.53823817f, 0.84270686f, 0.98932856f, 0.768665f, 0.90461975f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<op::v0::RNNCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9922437f, 0.97749525f, 0.9312212f, 0.9937176f, 0.9901317f, 0.95906746f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<op::v0::RNNCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94126844f, 0.9036043f, 0.841243f, 0.9468489f, 0.934215f, 0.873708f});

    test_case.run();
}
