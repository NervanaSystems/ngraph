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

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::v0::Elu>(A, 0.5f);
    auto function = make_shared<Function>(OutputVector{elu}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::v0::Elu>(A, -1.f);
    auto function = make_shared<Function>(OutputVector{elu}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
    test_case.run();
}
