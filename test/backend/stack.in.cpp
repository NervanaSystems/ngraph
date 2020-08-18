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

NGRAPH_TEST(${BACKEND_NAME}, stack_matrix_rowise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 2};
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_c);
    Shape shape_r{3, 2, 2};
    auto f = make_shared<Function>(make_shared<op::v0::Stack>(OutputVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 8});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 8, 8, 1, 2, 4, 8, 2, 3, 5, 7}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, stack_matrix_colwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 2};
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 2};
    auto f = make_shared<Function>(make_shared<op::v0::Stack>(OutputVector{A, B, C}, 1),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 8});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 1, 2, 2, 3, 8, 8, 4, 8, 5, 7}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, stack_negative_axis)
{
    auto pshape_a = PartialShape::dynamic();
    auto A = make_shared<op::v0::Parameter>(element::f32, pshape_a);
    auto pshape_b = PartialShape::dynamic();
    auto B = make_shared<op::v0::Parameter>(element::f32, pshape_b);
    auto pshape_c = PartialShape::dynamic();
    auto C = make_shared<op::v0::Parameter>(element::f32, pshape_c);
    auto pshape_r = PartialShape::dynamic();
    auto f = make_shared<Function>(make_shared<op::v0::Stack>(OutputVector{A, B, C}, -1),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{2, 2});
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2});
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto c = backend->create_tensor(element::f32, Shape{2, 2});
    copy_data(c, vector<float>{2, 3, 5, 7});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    ASSERT_EQ(result->get_shape(), (Shape{2, 2, 3}));
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 1, 2, 4, 2, 3, 8, 4, 5, 16, 8, 7}),
                                  read_vector<float>(result)));
}
