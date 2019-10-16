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

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"
#include "ngraph/ngraph.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx_${BACKEND_NAME}, utils_reshape_interpret_as_scalar_param)
{
    auto param_1d = make_shared<op::Parameter>(element::i64, Shape{1});
    auto scalar = onnx_import::reshape::interpret_as_scalar(param_1d);

    auto result = make_shared<op::Result>(scalar);
    auto function = make_shared<Function>(ResultVector{result}, ParameterVector{param_1d});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(std::vector<int64_t>{-42});
    test_case.add_expected_output<int64_t>(Shape{}, {-42});
    test_case.run();
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, utils_reshape_interpret_as_scalar_constant)
{
    auto constant_1d =
        make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{-42});
    auto scalar = onnx_import::reshape::interpret_as_scalar(constant_1d);

    auto result = make_shared<op::Result>(scalar);
    auto function = make_shared<Function>(ResultVector{result}, ParameterVector{});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_expected_output<int64_t>(Shape{}, {-42});
    test_case.run();
}
