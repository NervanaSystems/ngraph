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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid)
{
    const Shape shape{2, 7};
    const float alpha_f = 0.125f;
    const float beta_f = 0.642f;

    const auto A = make_shared<op::v0::Parameter>(element::f32, shape);

    const auto alpha = op::v0::Constant::create<float>(A->get_element_type(), Shape{}, {alpha_f});
    const auto beta = op::v0::Constant::create<float>(A->get_element_type(), Shape{}, {beta_f});

    auto hardsigmoid = make_shared<op::v0::HardSigmoid>(A, alpha, beta);
    auto f0 = make_shared<Function>(OutputVector{hardsigmoid}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Prepare input and expected output data
    vector<float> input_data{-1.f,
                             0.f,
                             1.f,
                             -100.f,
                             100.f,
                             -3.1234567f,
                             5.876543f,
                             7.13245364f,
                             numeric_limits<float>::max(),
                             numeric_limits<float>::lowest(),
                             numeric_limits<float>::min(),
                             numeric_limits<float>::infinity(),
                             numeric_limits<float>::min() / 16.f,
                             -numeric_limits<float>::min() / 16.f};

    auto impl = [alpha_f, beta_f](float val) { return min(max(alpha_f * val + beta_f, 0.f), 1.f); };
    vector<float> expected_output;
    transform(begin(input_data), end(input_data), back_inserter(expected_output), impl);

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, input_data);
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a});

    EXPECT_TRUE(test::all_close_f(expected_output, read_vector<float>(result0)));
}
