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

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_simple)
{
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{1, 15, 2, 2});
    auto tested_op = make_shared<op::v0::ShuffleChannels>(data, 1, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{1, 15, 2, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_negative_axis)
{
    // in this test the output is the same as in shuffle_channels_simple but
    // the axis value is negative and the C(channels) value is in a different dimension(0) of the
    // shape
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{15, 2, 1, 2});
    auto tested_op = make_shared<op::v0::ShuffleChannels>(data, -4, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{15, 2, 1, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_float)
{
    const auto data = make_shared<op::v0::Parameter>(element::f32, Shape{6, 1, 1, 1});
    auto tested_op = make_shared<op::v0::ShuffleChannels>(data, 0, 2);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    test_case.add_expected_output<float>(Shape{6, 1, 1, 1}, {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f});

    test_case.run();
}
