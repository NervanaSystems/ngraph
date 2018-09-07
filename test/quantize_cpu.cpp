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
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

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
    DequantizeTest<int8_t>(42, -1.0f, 300.0f, static_cast<float>(99.212601));
}
