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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close_f.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean)
{
    const Shape input_shape{4, 3, 2};
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto mean_builder = builder::opset1::mean(input, axes);

    auto f = make_shared<Function>(mean_builder, ParameterVector{input});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto handle = backend->compile(f);
    auto input_tensor = backend->create_tensor(element::f32, input_shape);
    vector<float> values(shape_size(input_shape));
    iota(begin(values), end(values), 0);
    copy_data(input_tensor, values);

    const Shape expected_output_shape{4};
    auto result = backend->create_tensor(element::f32, expected_output_shape);
    handle->call_with_validate({result}, {input_tensor});

    EXPECT_EQ(result->get_element_type(), input->get_element_type());
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    const vector<float> expected_output{2.5f, 8.5f, 14.5f, 20.5f};
    const auto output_values = read_vector<float>(result);
    EXPECT_EQ(output_values, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic)
{
    const AxisSet axes{0, 1};
    const auto input = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto mean_builder = builder::opset1::mean(input, axes);

    auto f = make_shared<Function>(mean_builder, ParameterVector{input});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto handle = backend->compile(f);

    const Shape input_shape{2, 4, 5};
    auto input_tensor = backend->create_tensor(element::f32, input_shape);
    vector<float> values(shape_size(input_shape));
    iota(begin(values), end(values), 0);
    copy_data(input_tensor, values);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    handle->call_with_validate({result}, {input_tensor});

    EXPECT_EQ(result->get_element_type(), input->get_element_type());
    const Shape expected_output_shape{5};
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    const vector<float> expected_output{17.5f, 18.5f, 19.5f, 20.5f, 21.5f};
    const auto output_values = read_vector<float>(result);
    EXPECT_EQ(output_values, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, builder_opset1_mean_dynamic_2)
{
    const AxisSet axes{1, 2};
    const auto input = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    const auto mean_builder = builder::opset1::mean(input, axes);

    auto f = make_shared<Function>(mean_builder, ParameterVector{input});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto handle = backend->compile(f);

    const Shape input_shape{2, 1, 3};
    auto input_tensor = backend->create_tensor(element::f32, input_shape);
    vector<float> values(shape_size(input_shape));
    iota(begin(values), end(values), 0);
    copy_data(input_tensor, values);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    handle->call_with_validate({result}, {input_tensor});

    EXPECT_EQ(result->get_element_type(), input->get_element_type());
    const Shape expected_output_shape{2};
    EXPECT_EQ(result->get_shape(), expected_output_shape);
    const vector<float> expected_output{1.f, 4.f};
    const auto output_values = read_vector<float>(result);
    EXPECT_EQ(output_values, expected_output);
}
