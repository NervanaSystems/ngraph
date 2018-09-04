/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "ngraph/op/constant.hpp"
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
    vector<float> a_data = {-85.0, 0.0, 2.0, 10.0, 15.0};
    Shape shape_a{5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {-85.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {15.0f});
    auto QT = make_shared<op::Quantize>(A, B, C, element::u8);
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
    auto result_min = backend->create_tensor(element::f32, Shape{});
    auto result_max = backend->create_tensor(element::f32, Shape{});
    backend->call(f, {result, result_min, result_max}, {a});
    EXPECT_EQ((vector<uint8_t>{0, 0, 6, 30, 45}), read_vector<uint8_t>(result));
    EXPECT_EQ((vector<float>{0.0}), read_vector<float>(result_min));
    EXPECT_EQ((vector<float>{85.0}), read_vector<float>(result_max));
}

TEST(quantize_cpu, quantize_to_uint8)
{
    vector<float> a_data = {-255.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0, 500.0};
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {-255.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto QT = make_shared<op::Quantize>(A, B, C, element::u8);
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
    auto result_min = backend->create_tensor(element::f32, Shape{});
    auto result_max = backend->create_tensor(element::f32, Shape{});
    backend->call(f, {result, result_min, result_max}, {a});
    EXPECT_EQ((vector<uint8_t>{0, 0, 1, 1, 2, 64, 127, 255}), read_vector<uint8_t>(result));
    EXPECT_EQ((vector<float>{0.0}), read_vector<float>(result_min));
    EXPECT_EQ((vector<float>{255.0}), read_vector<float>(result_max));
}

TEST(quantize_cpu, quantize_to_int8)
{
    vector<float> a_data = {-127.0, 0.0, 1.0, 3.0, 5.0, 64.0, 127.0, 500.0};
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {-127.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto QT = make_shared<op::Quantize>(A, B, C, element::i8);
    auto output_data = std::make_shared<op::GetOutputElement>(QT, 0);
    auto output_min = std::make_shared<op::GetOutputElement>(QT, 1);
    auto output_max = std::make_shared<op::GetOutputElement>(QT, 2);
    auto f = make_shared<Function>(NodeVector{output_data, output_min, output_max},
                                   op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_a);
    auto result_min = backend->create_tensor(element::f32, Shape{});
    auto result_max = backend->create_tensor(element::f32, Shape{});
    backend->call(f, {result, result_min, result_max}, {a});
    EXPECT_EQ((vector<int8_t>{-127, 0, 1, 3, 5, 64, 127, 127}), read_vector<int8_t>(result));
    EXPECT_EQ((vector<float>{-127}), read_vector<float>(result_min));
    EXPECT_EQ((vector<float>{127}), read_vector<float>(result_max));
}

template <typename T>
void DequantizeTest(int input, float min, float max, float expected_output)
{
    vector<T> a_data = {static_cast<T>(input)};
    Shape shape_a{1};
    auto A = make_shared<op::Parameter>(element::from<T>(), shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {min});
    auto C = op::Constant::create(element::f32, Shape{}, {max});
    auto r = make_shared<op::Dequantize>(A, B, C, element::from<T>());
    auto f = make_shared<Function>(r, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::from<T>(), Shape{});
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, Shape{});
    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{expected_output}), read_vector<float>(result));
}

TEST(quantize_cpu, dequantize_from_uint8)
{
    DequantizeTest<uint8_t>(255, 100.0f, 300.0f, 300.0);
}

TEST(quantize_cpu, dequantize_from_uint8_smallrange)
{
    DequantizeTest<uint8_t>(255, -2.0f, 5.0f, 5.0);
}

TEST(quantize_cpu, dequantize_from_int8_smallrange)
{
    DequantizeTest<int8_t>(-127, -2.0f, 1.0f, -2.0);
}

TEST(quantize_cpu, dequantize_from_int8)
{
    DequantizeTest<int8_t>(42, -1.0f, 300.0f, 99.212601);
}
