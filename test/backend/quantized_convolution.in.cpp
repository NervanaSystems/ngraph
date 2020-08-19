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

NGRAPH_TEST(${BACKEND_NAME}, quantized_convolution)
{
    Shape shape_a{1, 1, 3, 4};
    Shape shape_b{1, 1, 3, 3};
    Shape shape_r{1, 1, 3, 4};
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::v0::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i8, shape_b);
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto D = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto E = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto F = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto G = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto H = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto CV = ngraph::builder::QuantizedConvolutionBuilder(A,
                                                           B,
                                                           Strides{1, 1},
                                                           Strides{1, 1},
                                                           CoordinateDiff{1, 1},
                                                           CoordinateDiff{1, 1},
                                                           Strides{1, 1},
                                                           C,
                                                           D,
                                                           E,
                                                           F,
                                                           G,
                                                           H,
                                                           element::i8);
    auto f = make_shared<Function>(OutputVector{CV}, ParameterVector{A, B, C, D, E, F, G, H});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto d = backend->create_tensor(element::f32, Shape{});
    copy_data(d, vector<float>{0.0f});
    auto e = backend->create_tensor(element::f32, Shape{});
    copy_data(e, vector<float>{255.0f});
    auto e_a = backend->create_tensor(element::f32, Shape{});
    copy_data(e_a, vector<float>{-127.0f});
    auto g = backend->create_tensor(element::f32, Shape{});
    copy_data(g, vector<float>{127.0f});
    auto h = backend->create_tensor(element::f32, Shape{});
    copy_data(h, vector<float>{22.0f});
    auto i = backend->create_tensor(element::f32, Shape{});
    copy_data(i, vector<float>{90.0f});
    auto result = backend->create_tensor(element::i8, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, d, e, e_a, g, h, i});
    EXPECT_EQ((vector<int8_t>{31, 48, 42, 45, 54, 102, 127, 61, 47, 73, 61, 55}),
              read_vector<int8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, quantized_conv_int32_output)
{
    Shape shape_a{1, 1, 3, 4};
    Shape shape_b{1, 1, 3, 3};
    Shape shape_r{1, 1, 3, 4};
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<uint8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::v0::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::u8, shape_b);
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto D = op::v0::Constant::create(element::u8, Shape{}, {0});
    auto E = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto F = op::v0::Constant::create(element::u8, Shape{}, {0});
    auto G = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto H = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto CV = make_shared<op::v0::QuantizedConvolution>(A,
                                                        B,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{1, 1},
                                                        CoordinateDiff{1, 1},
                                                        Strides{1, 1},
                                                        C,
                                                        D,
                                                        E,
                                                        F,
                                                        G,
                                                        H,
                                                        element::i32);
    auto f = make_shared<Function>(OutputVector{CV}, ParameterVector{A, B, C, E, G});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::u8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, Shape{});
    copy_data(c, vector<float>{1.0f});
    auto d = backend->create_tensor(element::f32, Shape{});
    copy_data(d, vector<float>{1.0f});
    auto e = backend->create_tensor(element::f32, Shape{});
    copy_data(e, vector<float>{1.0f});
    auto result = backend->create_tensor(element::i32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c, d, e});
    EXPECT_EQ((vector<int32_t>{22, 34, 30, 32, 38, 72, 90, 43, 33, 52, 43, 39}),
              read_vector<int32_t>(result));
}
