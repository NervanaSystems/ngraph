//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dyn_group_convolution_backprop_data)
{
    Shape shape_filter{6, 3, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_data_batch_shape{2, 3, 5, 5};
    auto data_batch_shape =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};

    auto conv1 = make_shared<op::v1::ConvolutionBackpropData>(
        filters, deltas, data_batch_shape, strides, dilations, padding_begin, padding_end);

    auto f = make_shared<Function>(conv1, ParameterVector{filters, deltas, data_batch_shape});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> filter, delta, expected_result;

    for (int i = 0; i < 6 * 3 * 3 * 3; i++)
        filter.emplace_back(i);

    for (int i = 0; i < 2 * 6 * 3 * 3; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        expected_result.emplace_back(i);

    vector<int64_t> shapes = {2, 3, 5, 5};

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_filter);
    copy_data(a, filter);
    auto b = backend->create_tensor(element::f32, shape_delta);
    copy_data(b, delta);
    auto c = backend->create_tensor(element::i64, Shape{shapes.size()}); // dynamic data batch shape
    copy_data(c, shapes);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dyn_group_convolution_backprop_filter)
{
    Shape shape_data{64, 3, 100};
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_delta{64, 128, 96};
    auto deltas = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto filters_shape =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = Strides{1};
    auto dilations = Strides{1};
    auto padding_begin = CoordinateDiff{2};
    auto padding_end = CoordinateDiff{3};
    auto conv1 = make_shared<op::v1::ConvolutionBackpropFilters>(
        data, deltas, filters_shape, strides, dilations, padding_begin, padding_end);

    auto f = make_shared<Function>(conv1, ParameterVector{data, deltas, filters_shape});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> input, delta, expected_result;

    for (int i = 0; i < 64 * 3 * 100; i++)
        input.emplace_back(i);

    for (int i = 0; i < 64 * 128 * 96; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 128 * 3 * 10; i++)
        expected_result.emplace_back(i);

    vector<int64_t> shapes = {128, 3, 10};

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_data);
    copy_data(a, input);
    auto b = backend->create_tensor(element::f32, shape_delta);
    copy_data(b, delta);
    auto c = backend->create_tensor(element::i64, Shape{shapes.size()}); // dynamic data batch shape
    copy_data(c, shapes);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}
