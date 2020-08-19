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

NGRAPH_TEST(${BACKEND_NAME}, gemm)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});

    auto gemm_func = make_shared<op::v0::Gemm>(A, B, C);
    auto function = make_shared<Function>(OutputVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>(12, 0));
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 12));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_C)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});

    auto gemm_func = make_shared<op::v0::Gemm>(A, B, C);
    auto function = make_shared<Function>(OutputVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>(12, 1));
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 13));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_broadcast_input_C)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{});

    auto gemm_func = make_shared<op::v0::Gemm>(A, B, C, 0.5);
    auto function = make_shared<Function>(OutputVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>{1});
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 7));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_broadcast_axes_0_input_C)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});

    auto gemm_func = make_shared<op::v0::Gemm>(A, B, C, 0.5);
    auto function = make_shared<Function>(OutputVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>{1, 2, 3, 4});
    // output
    test_case.add_expected_output<float>(Shape{3, 4},
                                         vector<float>{7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_broadcast_axes_1_input_C)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{3, 1});

    auto gemm_func = make_shared<op::v0::Gemm>(A, B, C, 0.5);
    auto function = make_shared<Function>(OutputVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>(3, 1));
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 7));
    test_case.run();
}
