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
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/test_vector_generator.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

#define NGRAPH_TYPED_TEST(c_func__, op__, min__, max__)                                            \
    NGRAPH_TEST(${BACKEND_NAME}, c_func__##_${DATA_TYPE})                                          \
    {                                                                                              \
        auto input = test::make_floating_point_data<${DATA_TYPE}>(min__, max__);                   \
        auto output = vector<${DATA_TYPE}>(input.size());                                          \
        Shape shape = Shape{input.size()};                                                         \
        auto A = make_shared<op::Parameter>(${ELEMENT_TYPE}, shape);                               \
        auto f = make_shared<Function>(make_shared<op__>(A), ParameterVector{A});                  \
                                                                                                   \
        auto backend = runtime::Backend::create("${BACKEND_NAME}");                                \
        auto a = backend->create_tensor(${ELEMENT_TYPE}, shape);                                   \
        auto result = backend->create_tensor(${ELEMENT_TYPE}, shape);                              \
                                                                                                   \
        copy_data(a, input);                                                                       \
                                                                                                   \
        auto handle = backend->compile(f);                                                         \
        handle->call_with_validate({result}, {a});                                                 \
        auto output_data = read_vector<${DATA_TYPE}>(result);                                      \
                                                                                                   \
        vector<${DATA_TYPE}> expected;                                                             \
        for (auto x : input)                                                                       \
        {                                                                                          \
            expected.push_back(c_func__(x));                                                       \
        }                                                                                          \
                                                                                                   \
        EXPECT_TRUE(test::all_close_f(expected, output_data, MIN_FLOAT_TOLERANCE_BITS));           \
    }

// clang-format off
static ${DATA_TYPE} negative(${DATA_TYPE} x) { return -x;}
static ${DATA_TYPE} sign(${DATA_TYPE} x) { return x==0 ? 0 : (x>0 ? 1 : -1);}
NGRAPH_TYPED_TEST(abs, ngraph::op::Abs, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(acos, ngraph::op::Acos, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(asin, ngraph::op::Asin, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(atan, ngraph::op::Atan, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(ceil, ngraph::op::Ceiling, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(cos, ngraph::op::Cos, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(cosh, ngraph::op::Cosh, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(erf, ngraph::op::Erf, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(exp, ngraph::op::Exp, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(floor, ngraph::op::Floor, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(log, ngraph::op::Log, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(negative, ngraph::op::Negative, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(sign, ngraph::op::Sign, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(sin, ngraph::op::Sin, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(sinh, ngraph::op::Sinh, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(sqrt, ngraph::op::Sqrt, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(tan, ngraph::op::Tan, -1e+10, 1e+10);
NGRAPH_TYPED_TEST(tanh, ngraph::op::Tanh, -1e+10, 1e+10);
// clang-format on
