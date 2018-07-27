/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/cpu/op/dequantize.hpp"
#include "ngraph/runtime/cpu/op/quantize.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(quantize_cpu, quantize_to_uint8_small)
{
    vector<float> a_data = {-1.0, 0.0, 2.0};
    const float input_min = -1.0f;
    const float input_max = 2.0f;
    Shape shape_a{3};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto QT = make_shared<op::Quantize>(A, input_min, input_max);
    auto output_data = std::make_shared<op::GetOutputElement>(QT, 0);
    auto output_min = std::make_shared<op::GetOutputElement>(QT, 1);
    auto output_max = std::make_shared<op::GetOutputElement>(QT, 2);

    auto f = make_shared<Function>(NodeVector{output_data, output_min, output_max},
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = backend->create_tensor(element::u8, shape_a);
    auto result_min = backend->create_tensor(element::f32, Shape{1});
    auto result_max = backend->create_tensor(element::f32, Shape{1});

    backend->call(f, {result, result_min, result_max}, {a});

    EXPECT_EQ((vector<uint8_t>{0, 0, 255}), read_vector<uint8_t>(result));
    EXPECT_EQ((vector<float>{0.0}), read_vector<float>(result_min));
    EXPECT_EQ((vector<float>{2.0}), read_vector<float>(result_max));
}

TEST(quantize_cpu, quantize_to_uint8)
{
    vector<float> a_data = {-255.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0, 500.0};
    const float input_min = -255.0f;
    const float input_max = 127.0f;
    Shape shape_a{8};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto QT = make_shared<op::Quantize>(A, input_min, input_max);
    auto output_data = std::make_shared<op::GetOutputElement>(QT, 0);
    auto output_min = std::make_shared<op::GetOutputElement>(QT, 1);
    auto output_max = std::make_shared<op::GetOutputElement>(QT, 2);

    auto f = make_shared<Function>(NodeVector{output_data, output_min, output_max},
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = backend->create_tensor(element::u8, shape_a);
    auto result_min = backend->create_tensor(element::f32, Shape{1});
    auto result_max = backend->create_tensor(element::f32, Shape{1});

    backend->call(f, {result, result_min, result_max}, {a});

    EXPECT_EQ((vector<uint8_t>{0, 0, 1, 1, 2, 64, 127, 255}), read_vector<uint8_t>(result));
    EXPECT_EQ((vector<float>{0.0}), read_vector<float>(result_min));
    EXPECT_EQ((vector<float>{255.0}), read_vector<float>(result_max));
}

TEST(quantize_cpu, dequantize_from_uint8_scale_up)
{
    vector<uint8_t> a_data = {255};
    const float input_min = 200.0f;
    const float input_max = 400.0f;
    Shape shape_a{1};

    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto r = make_shared<op::Dequantize>(A, input_min, input_max);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, Shape{1});
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, Shape{1});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{400.0}), read_vector<float>(result));
}

TEST(quantize_cpu, dequantize_from_uint8_scale_down)
{
    vector<uint8_t> a_data = {255};
    const float input_min = -1.0f;
    const float input_max = 2.0f;
    Shape shape_a{1};

    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto r = make_shared<op::Dequantize>(A, input_min, input_max);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, Shape{1});
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, Shape{1});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2.0}), read_vector<float>(result));
}
