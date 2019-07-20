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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, tensorview_custom_mem)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

        return f;
    };

    auto f = make_external();

    vector<float> av{2, 4, 8, 16};
    vector<float> bv{1, 2, 4, 8};
    // use custom mem with tensorview, no need to copy data
    auto a = backend->create_tensor(element::f32, shape, av.data());
    auto b = backend->create_tensor(element::f32, shape, bv.data());

    // use custom mem with result tensorview
    vector<float> rv{0, 0, 0, 0};
    auto result = backend->create_tensor(element::f32, shape, rv.data());

    // result should be in memory without needing explict read
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 2}), rv, MIN_FLOAT_TOLERANCE_BITS));
}
