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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, lrn)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    vector<float> expected{0.f,
                           0.3015113f,
                           0.4364357f,
                           0.5f,
                           0.8728715f,
                           0.8451542f,
                           0.5970223f,
                           0.6115928f,
                           0.5642765f,
                           0.5669467f,
                           0.7784989f,
                           0.7720487f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}
