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

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, max_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{1, 2, 3, 4}), read_vector<int8_t>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, max_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_5d_int32)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{4}), read_vector<float>(result)));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{4}), read_vector<int8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6}), read_vector<float>(result)));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 6}), read_vector<float>(result)));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_int32)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{2, 4, 6}), read_vector<int32_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int32_t>{1, 2, 3, 4, 5, 6}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(
        test::all_close_f((vector<float>{}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_int32)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{});
    auto result = backend->create_tensor(element::i32, shape_rt);
    copy_data(result, vector<int32_t>({3, 3, 3}));

    int32_t minval = std::numeric_limits<int32_t>::has_infinity
                         ? -std::numeric_limits<int32_t>::infinity()
                         : std::numeric_limits<int32_t>::min();

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{minval, minval, minval}), read_vector<int32_t>(result));
    EXPECT_EQ((vector<int32_t>{}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(
        test::all_close_f((vector<float>{}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(
        test::all_close_f((vector<float>{}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_TRUE(
        test::all_close_f((vector<float>{}), read_vector<float>(a), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{25.0f, 26.0f, 27.0f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{14.0f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{14}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_double)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::v0::Parameter>(element::f64, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_a);
    copy_data(a, vector<double>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f64, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<double>{14}), read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::v0::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float mi = -std::numeric_limits<float>::infinity();

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<float>{mi, mi, mi, mi, mi, mi}), read_vector<float>(result));
}
