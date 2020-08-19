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

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = false;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto gru_cell = make_shared<op::v3::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid", "tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f});

    // Ht
    test_case.add_input<float>(
        {0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f});

    // W
    test_case.add_input<float>(
        {0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
         0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
         0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
         0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f});

    // R
    test_case.add_input<float>(
        {0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f, 0.1623331f,  0.01389796f, 0.99669236f,
         0.53901845f, 0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f, 0.40111724f, 0.37497616f,
         0.2903471f,  0.6796794f,  0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f, 0.791677f,
         0.76497287f, 0.9895242f,  0.7845312f,  0.51267904f, 0.49030215f, 0.08498167f});

    // B (the sum of biases for W and R)
    test_case.add_input<float>({
        0.8286678f + 0.9175602f,
        0.9153158f + 0.14958014f,
        0.9581612f + 0.49230585f,
        0.6639213f + 0.63162816f,
        0.84239805f + 0.4161903f,
        0.5282445f + 0.22148274f,
        0.14153397f + 0.50496656f,
        0.22404431f + 0.34798595f,
        0.6549655f + 0.6699164f,
    });

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.48588726f, 0.99670005f, 0.83759373f, 0.5023099f, 0.89410484f, 0.60011315f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_linear_before_reset)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B =
        make_shared<op::v0::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<op::v3::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid", "tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});
    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.61395123f, // 0.09875853f + 0.5151927f,
                                1.08667738f, // 0.37801138f + 0.708666f,
                                1.32600244f, // 0.7729636f + 0.55303884f,
                                0.81917698f, // 0.78493553f + 0.03424145f,
                                1.37736335f, // 0.5662702f + 0.81109315f,
                                0.42931147f, // 0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8709214f, 0.48411977f, 0.74495184f, 0.6074972f, 0.44572943f, 0.1467715f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::v0::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B =
        make_shared<op::v0::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<op::v3::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"hardsigmoid", "hardsigmoid"},
                                                       vector<float>{1.8345f, 1.8345f},
                                                       vector<float>{3.05f, 3.05f},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});

    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.09875853f + 0.5151927f,
                                0.37801138f + 0.708666f,
                                0.7729636f + 0.55303884f,
                                0.78493553f + 0.03424145f,
                                0.5662702f + 0.81109315f,
                                0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    test_case.run();
}
