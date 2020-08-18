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

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes =
        make_shared<op::v0::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01428571f, 0.02857143f, 0.04285714f, 0.05714286f, 0.07142857f, 0.08571429f,
                     0.1f,        0.11428571f, 0.12857144f, 0.14285715f, 0.15714286f, 0.17142858f,
                     0.18571429f, 0.2f,        0.21428572f, 0.22857143f, 0.24285714f, 0.25714287f,
                     0.27142859f, 0.2857143f,  0.30000001f, 0.31428573f, 0.32857144f, 0.34285715f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_empty_axes_input)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::v0::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    // output should be filled with 1f values
    test_case.add_expected_output<float>(data_shape, vector<float>(shape_size(data_shape), 1));

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_h_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_1axis_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_123axes_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes =
        make_shared<op::v0::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.02638899f, 0.04956816f, 0.070014f,   0.10555596f, 0.1239204f,  0.140028f,
                     0.18472293f, 0.19827265f, 0.210042f,   0.26388991f, 0.27262488f, 0.280056f,
                     0.34305686f, 0.34697714f, 0.35007f,    0.42222384f, 0.42132938f, 0.420084f,
                     0.50139081f, 0.49568161f, 0.49009803f, 0.58055776f, 0.57003385f, 0.560112f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_c_2x2_shape)
{
    Shape data_shape{2, 2};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::v0::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.44721353f, 0.89442706f, 0.60000002f, 0.80000001f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_c_2x4_shape)
{
    Shape data_shape{2, 4};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::v0::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.18257418f,
                                          0.36514837f,
                                          0.54772252f,
                                          0.73029673f,
                                          0.37904903f,
                                          0.45485884f,
                                          0.53066862f,
                                          0.60647845f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d_max_bias)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto axes =
        make_shared<op::v0::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{5000};
    auto eps_mode = op::EpsMode::MAX;

    auto normalize = make_shared<op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(OutputVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01414214f, 0.02828427f, 0.04242641f, 0.05656854f, 0.07071068f, 0.08485281f,
                     0.09899495f, 0.11313709f, 0.12727922f, 0.14142136f, 0.15556349f, 0.16970563f,
                     0.18384777f, 0.1979899f,  0.21213204f, 0.22627418f, 0.2404163f,  0.25455844f,
                     0.26870057f, 0.28284273f, 0.29698485f, 0.31112698f, 0.32526913f, 0.33941126f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}
