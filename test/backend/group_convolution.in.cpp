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
    Shape shape_filter{6, 1, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_data_batch{2, 3, 5, 5};
    auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};
    size_t groups = 3;

    auto conv_bprop_data = make_shared<op::GroupConvolutionBackpropData>(
        data_batch, filters, deltas, strides, dilations, padding_begin, padding_end, groups);

    auto f = make_shared<Function>(conv_bprop_data, ParameterVector{data_batch, filters, deltas});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> filter, delta, data, expected_result;

    for (int i = 0; i < 6 * 1 * 3 * 3; i++)
        filter.emplace_back(i);

    for (int i = 0; i < 2 * 6 * 3 * 3; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        data.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        expected_result.emplace_back(i);

    auto a = backend->create_tensor(element::f32, shape_data_batch);
    copy_data(a, data);
    auto b = backend->create_tensor(element::f32, shape_filter);
    copy_data(b, filter);
    auto c = backend->create_tensor(element::f32, shape_delta);
    copy_data(c, delta);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, dyn_group_convolution_backprop_filters)
{
    Shape shape_filter{6, 1, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape shape_data_batch{2, 3, 5, 5};
    auto data_batch = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};
    size_t groups = 3;

    auto conv_bprop_filters = make_shared<op::GroupConvolutionBackpropFilters>(
        data_batch, filters, deltas, strides, dilations, padding_begin, padding_end, groups);

    auto f =
        make_shared<Function>(conv_bprop_filters, ParameterVector{data_batch, filters, deltas});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> filter, delta, data, expected_result;

    for (int i = 0; i < 6 * 1 * 3 * 3; i++)
        filter.emplace_back(i);

    for (int i = 0; i < 2 * 6 * 3 * 3; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        data.emplace_back(i);

    for (int i = 0; i < 6 * 1 * 3 * 3; i++)
        expected_result.emplace_back(i);

    auto a = backend->create_tensor(element::f32, shape_data_batch);
    copy_data(a, data);
    auto b = backend->create_tensor(element::f32, shape_filter);
    copy_data(b, filter);
    auto c = backend->create_tensor(element::f32, shape_delta);
    copy_data(c, delta);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}
