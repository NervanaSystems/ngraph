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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/float_util.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/test_vector_generator.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

#define NGRAPH_TYPED_BINARY_TEST(name__, op__, validator__)                                        \
    NGRAPH_TEST(${BACKEND_NAME}, name__##_${DATA_TYPE})                                            \
    {                                                                                              \
        auto input0 = test::make_floating_point_data<${DATA_TYPE}>();                              \
        auto input1 = test::make_floating_point_data<${DATA_TYPE}>();                              \
        Shape shape = Shape{input0.size()};                                                        \
        auto A = make_shared<op::Parameter>(element::f32, shape);                                  \
        auto B = make_shared<op::Parameter>(element::f32, shape);                                  \
        auto f = make_shared<Function>(make_shared<op__>(A, B), ParameterVector{A, B});            \
                                                                                                   \
        auto backend = runtime::Backend::create("${BACKEND_NAME}");                                \
                                                                                                   \
        shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);               \
        shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);               \
        shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);          \
                                                                                                   \
        copy_data(a, input0);                                                                      \
        copy_data(b, input1);                                                                      \
                                                                                                   \
        auto handle = backend->compile(f);                                                         \
        handle->call_with_validate({result}, {a, b});                                              \
        auto output_data = read_vector<${DATA_TYPE}>(result);                                      \
                                                                                                   \
        vector<${DATA_TYPE}> expected;                                                             \
        for (size_t i = 0; i < input0.size(); i++)                                                 \
        {                                                                                          \
            expected.push_back(validator__);                                                       \
        }                                                                                          \
                                                                                                   \
        EXPECT_TRUE(test::all_close_f(expected, output_data, MIN_FLOAT_TOLERANCE_BITS));           \
    }

NGRAPH_TYPED_BINARY_TEST(add, ngraph::op::Add, (input0[i] + input1[i]));
NGRAPH_TYPED_BINARY_TEST(subtract, ngraph::op::Subtract, (input0[i] - input1[i]));
NGRAPH_TYPED_BINARY_TEST(multiply, ngraph::op::Multiply, (input0[i] * input1[i]));
NGRAPH_TYPED_BINARY_TEST(divide, ngraph::op::Divide, (input0[i] / input1[i]));
NGRAPH_TYPED_BINARY_TEST(minimum,
                         ngraph::op::Minimum,
                         (input0[i] < input1[i] ? input0[i] : input1[i]));
NGRAPH_TYPED_BINARY_TEST(maximum,
                         ngraph::op::Maximum,
                         (input0[i] > input1[i] ? input0[i] : input1[i]));
NGRAPH_TYPED_BINARY_TEST(power, ngraph::op::Power, (pow(input0[i], input1[i])));
