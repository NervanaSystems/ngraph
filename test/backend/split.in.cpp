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

NGRAPH_TEST(${BACKEND_NAME}, split_3_equal_parts)
{
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{6});
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});

    const auto tested_op = make_shared<op::v0::Split>(data, axis, 3);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6});

    test_case.add_expected_output<int32_t>(Shape{2}, {1, 2});
    test_case.add_expected_output<int32_t>(Shape{2}, {3, 4});
    test_case.add_expected_output<int32_t>(Shape{2}, {5, 6});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_var_len_parts)
{
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{2, 6});

    const std::vector<size_t> splits = {2, 4};
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    const auto tested_op = make_shared<op::v0::Split>(data, axis, splits);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 1, 6, 7});
    test_case.add_expected_output<int32_t>(Shape{2, 4}, {2, 3, 4, 5, 8, 9, 10, 11});

    test_case.run();
}
