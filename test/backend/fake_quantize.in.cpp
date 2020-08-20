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

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 4;
    const auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto quantize = make_shared<op::v0::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        OutputVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({0.0f});
    // input_high
    test_case.add_input<float>({23.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,          2.f,          2.f,          2.f,          6.6666669f,
                      6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                      6.6666669f,   6.6666669f,   11.33333301f, 11.33333301f, 11.33333301f,
                      11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                      16.f,         16.f,         16.f,         16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 5;
    const auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto quantize = make_shared<op::v0::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        OutputVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({3.f});
    // input_high
    test_case.add_input<float>({17.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                      12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip_across_channels)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 1});
    auto input_high = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_low = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 1});
    auto output_high = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 1});

    auto quantize = make_shared<op::v0::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels);
    auto function = make_shared<Function>(
        OutputVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_pdpd)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto input_high = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto output_low = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto output_high = make_shared<op::v0::Parameter>(element::f32, Shape{2});

    auto quantize =
        make_shared<op::v0::FakeQuantize>(data,
                                          input_low,
                                          input_high,
                                          output_low,
                                          output_high,
                                          levels,
                                          op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    auto function = make_shared<Function>(
        OutputVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}
