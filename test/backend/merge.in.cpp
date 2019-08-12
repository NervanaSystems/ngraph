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
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, merge_output_true)
{
    Shape cond_shape{};
    Shape tval_shape{6};
    Shape fval_shape{6};
    Shape rshape{6};
    auto C = make_shared<op::Parameter>(element::boolean, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f32, fval_shape);
    auto R = make_shared<Function>(make_shared<op::Merge>(C, T, F), ParameterVector{C, T, F});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto c = backend->create_tensor(element::boolean, cond_shape);
    copy_data(c, vector<char>{1});
    auto t = backend->create_tensor(element::f32, tval_shape);
    copy_data(t, vector<float>{1, 2, 3, 4, 5, 6});
    auto f = backend->create_tensor(element::f32, fval_shape);
    copy_data(f, vector<float>{-1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f32, rshape);

    auto h = backend->compile(R);
    h->call_with_validate({result}, {c, t, f});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, merge_output_false)
{
    Shape cond_shape{};
    Shape tval_shape{6};
    Shape fval_shape{6};
    Shape rshape{6};
    auto C = make_shared<op::Parameter>(element::boolean, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f32, fval_shape);
    auto R = make_shared<Function>(make_shared<op::Merge>(C, T, F), ParameterVector{C, T, F});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto c = backend->create_tensor(element::boolean, cond_shape);
    copy_data(c, vector<char>{0});
    auto t = backend->create_tensor(element::f32, tval_shape);
    copy_data(t, vector<float>{1, 2, 3, 4, 5, 6});
    auto f = backend->create_tensor(element::f32, fval_shape);
    copy_data(f, vector<float>{-1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f32, rshape);

    auto h = backend->compile(R);
    h->call_with_validate({result}, {c, t, f});
    EXPECT_TRUE(test::all_close_f((vector<float>{-1, -2, -3, -4, -5, -6}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
