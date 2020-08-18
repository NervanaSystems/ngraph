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

#include "ngraph/ngraph.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;
using namespace ngraph::test;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean)
{
    const Shape input_shape{4, 3, 2};
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{4}, vector<float>{2.5f, 8.5f, 14.5f, 20.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic)
{
    const Shape input_shape{2, 4, 5};
    const AxisSet axes{0, 1};
    const auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{5},
                                         vector<float>{17.5f, 18.5f, 19.5f, 20.5f, 21.5f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic_2)
{
    const Shape input_shape{2, 1, 3};
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);
    auto function = make_shared<Function>(mean_builder, ParameterVector{input});

    auto test_case = NgraphTestCase(function, "${BACKEND_NAME}", BackendMode::DYNAMIC);
    vector<float> input_values(shape_size(input_shape));
    iota(begin(input_values), end(input_values), 0);
    test_case.add_input<float>(input_shape, input_values);
    test_case.add_expected_output<float>(Shape{2}, vector<float>{1.f, 4.f});

    test_case.run();
}
