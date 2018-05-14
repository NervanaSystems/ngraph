/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

static const vector<element::Type> s_known_element_types = {element::from<float>(),
                                                            element::from<double>(),
                                                            element::from<int8_t>(),
                                                            element::from<int16_t>(),
                                                            element::from<int32_t>(),
                                                            element::from<int64_t>(),
                                                            element::from<uint8_t>(),
                                                            element::from<uint16_t>(),
                                                            element::from<uint32_t>(),
                                                            element::from<uint64_t>()};

NGRAPH_TEST(${BACKEND_NAME}, function_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B}, "funky func name");

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor<float>(shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor<float>(shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, node_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    C->set_name("a node name");
    auto f = make_shared<Function>(C, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    auto D = A * B;
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out1 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out2 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out3 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out4 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out5 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out6 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> out7 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    backend->call(f, {out1, out2, out3, out4, out5, out6, out7}, {a, b});
    EXPECT_EQ(expectedC, read_vector<float>(out1));
    EXPECT_EQ(expectedC, read_vector<float>(out2));
    EXPECT_EQ(expectedD, read_vector<float>(out3));
    EXPECT_EQ(expectedD, read_vector<float>(out4));
    EXPECT_EQ(expectedC, read_vector<float>(out5));
    EXPECT_EQ(expectedE, read_vector<float>(out6));
    EXPECT_EQ(expectedE, read_vector<float>(out7));
}

NGRAPH_TEST(${BACKEND_NAME}, parameter_as_output)
{
    Shape shape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    vector<float> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> zero(shape_size(shape), 0);
    copy_data(a, expected);

    backend->call(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, ab)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, abc_int64)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto C = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::i64, shape);
    copy_data(c, vector<int64_t>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call(f, {result}, {b, a, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call(f, {result}, {a, c, b});
    EXPECT_EQ((vector<int64_t>{50, 72, 98, 128}), read_vector<int64_t>(result));
}

// Multiple retrive values
NGRAPH_TEST(${BACKEND_NAME}, multiple_result)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto f =
        make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->create_tensor(element::f32, shape);
    auto r1 = backend->create_tensor(element::f32, shape);

    backend->call(f, {r0, r1}, {a, b, c});

    EXPECT_EQ((vector<float>{6, 8, 10, 12}), read_vector<float>(r0));
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(r1));
}

NGRAPH_TEST(${BACKEND_NAME}, abs)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 0, 4.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_one_output)
{
    auto shape_in = Shape{2, 3};
    auto shape_mean = Shape{3};

    auto A = make_shared<op::Parameter>(element::f64, shape_in);
    auto Mean =
        op::Constant::create(element::f64, shape_mean, {0.00396654, -1.25294404, 1.16651872});
    auto Variance =
        op::Constant::create(element::f64, shape_mean, {2.40871689, 1.44969511, 0.23469392});
    auto Beta =
        op::Constant::create(element::f64, shape_mean, {2.14211921, -0.75733924, 0.42210531});
    auto Gamma =
        op::Constant::create(element::f64, shape_mean, {1.75437676, 0.37950502, 1.13727544});

    auto BN = make_shared<op::BatchNorm>(1e-3, Gamma, Beta, A, Mean, Variance, false);
    auto f = make_shared<Function>(BN, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_in);
    copy_data(
        a,
        vector<double>{-1.97431703, -2.06521307, 0.54122217, 2.53375939, -0.22342691, 0.45340773});

    auto result = backend->create_tensor(element::f64, shape_in);
    vector<double> expected_result{
        -0.09365749, -1.01327395, -1.04269195, 5.00118923, -0.43295258, -1.24840283};

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(vector<double>{expected_result}, read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_three_outputs)
{
    auto shape_in = Shape{2, 3};
    auto shape_mean = Shape{3};

    auto A = make_shared<op::Parameter>(element::f64, shape_in);
    auto Beta =
        op::Constant::create(element::f64, shape_mean, {2.14211921, -0.75733924, 0.42210531});
    auto Gamma =
        op::Constant::create(element::f64, shape_mean, {1.75437676, 0.37950502, 1.13727544});

    auto BN = make_shared<op::BatchNorm>(1e-3, Gamma, Beta, A);

    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(BN, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(BN, 1), op::ParameterVector{A});
    auto f2 =
        make_shared<Function>(make_shared<op::GetOutputElement>(BN, 2), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_in);
    copy_data(
        a,
        vector<double>{-1.97431703, -2.06521307, 0.54122217, 2.53375939, -0.22342691, 0.45340773});

    auto result0 = backend->create_tensor(element::f64, shape_in);
    vector<double> expected_result0{
        0.3879149, -1.13662076, 1.34494817, 3.89632344, -0.37805778, -0.50073695};

    backend->call(f0, {result0}, {a});
    EXPECT_TRUE(test::all_close(vector<double>{expected_result0}, read_vector<double>(result0)));

    auto result1 = backend->create_tensor(element::f64, shape_mean);
    vector<double> expected_result1{0.27972114, -1.14431989, 0.49731493};

    backend->call(f1, {result1}, {a});
    EXPECT_TRUE(test::all_close(vector<double>{expected_result1}, read_vector<double>(result1)));

    auto result2 = backend->create_tensor(element::f64, shape_mean);
    vector<double> expected_result2{5.08068895e+00, 8.48043919e-01, 1.92784308e-03};

    backend->call(f2, {result2}, {a});
    EXPECT_TRUE(test::all_close(vector<double>{expected_result2}, read_vector<double>(result2)));
}

NGRAPH_TEST(${BACKEND_NAME}, ceiling)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 1),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_int64)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::i64, shape_c);
    copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::i64, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_vector)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor)
{
    Shape shape{1, 1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1, 1, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
NGRAPH_TEST(${BACKEND_NAME}, concat_5d)
{
    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    Shape shape_a{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 9, 3, 2};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto Y_out = f->get_output_op(0);
        auto Xs = f->get_parameters();
        auto C = std::make_shared<op::Parameter>(Y_out->get_element_type(), Y_out->get_shape());
        ngraph::autodiff::Adjoints adjoints(NodeVector{Y_out}, NodeVector{C});
        std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
        transform(
            Xs.begin(), Xs.end(), dYdXs.begin(), [C, &adjoints](const std::shared_ptr<Node>& X) {
                return adjoints.backprop_node(X);
            });
        std::vector<std::shared_ptr<op::Parameter>> params(Xs);
        params.push_back(C);

        auto bf = std::make_shared<Function>(dYdXs, params);

        return bf;
    };

    auto bf = make_external();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 2, 2, 2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{1, 1, 1, 1});

    auto resulta = backend->create_tensor(element::f32, shape);
    auto resultb = backend->create_tensor(element::f32, shape);

    backend->call(bf, {resulta, resultb}, {a, b, c});
    EXPECT_EQ((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta));
    EXPECT_EQ((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::i32, shape);

    EXPECT_ANY_THROW({ backend->call(f, {result}, {a, b}); });
}

NGRAPH_TEST(${BACKEND_NAME}, equal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_0_0)
{
    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112});

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_2x0_0x2)
{
    Shape shape_a{2, 0};
    Shape shape_b{0, 2};
    Shape shape_r{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{0, 0, 0, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_0x2_2x0)
{
    Shape shape_a{0, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_3x2_2x0)
{
    Shape shape_a{3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_0x2)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_2x0_0)
{
    Shape shape_a{2, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112});

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot1d)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{170}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot2d)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{19, 22, 43, 50}), read_vector<float>(result));
}

//
// Here is what numpy does:
//
// >>> a = linspace(1,2*2*2,2*2*2)
// >>> b = linspace(1,2*2*2,2*2*2)
//
// >>> a.shape=(2,2,2)
// >>> b.shape=(2,2,2)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[[ 11.,  14.],
//          [ 17.,  20.]],
//
//         [[ 23.,  30.],
//          [ 37.,  44.]]],
//
//
//        [[[ 35.,  46.],
//          [ 57.,  68.]],
//
//         [[ 47.,  62.],
//          [ 77.,  92.]]]])
//
NGRAPH_TEST(${BACKEND_NAME}, dot3d_3d)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92}),
              read_vector<float>(result));
}

//
// Here is what numpy does:
//
// >>> from numpy import *
// >>> a = linspace(0,4*2*3-1,4*2*3)
// >>> b = linspace(0,3*4-1,3*4)
//
// >>> a.shape=(4,2,3)
// >>> b.shape=(3,4)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[  20.,   23.,   26.,   29.],
//         [  56.,   68.,   80.,   92.]],
//
//        [[  92.,  113.,  134.,  155.],
//         [ 128.,  158.,  188.,  218.]],
//
//        [[ 164.,  203.,  242.,  281.],
//         [ 200.,  248.,  296.,  344.]],
//
//        [[ 236.,  293.,  350.,  407.],
//         [ 272.,  338.,  404.,  470.]]])
//
NGRAPH_TEST(${BACKEND_NAME}, dot3d_2d)
{
    Shape shape_a{4, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 2, 4};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134,
                             155, 128, 158, 188, 218, 164, 203, 242, 281, 200, 248,
                             296, 344, 236, 293, 350, 407, 272, 338, 404, 470}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_tensor_arg0)
{
    Shape shape_a{};
    Shape shape_b{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_b);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_tensor_arg1)
{
    Shape shape_a{2, 2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_a);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_scalar_scalar)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{8});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{48}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{190, 486, 782, 1078}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, dot_matrix_vector_int64)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{17, 18, 19, 20});
    auto result = backend->create_tensor(element::i64, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<int64_t>{190, 486, 782, 1078}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greater)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greatereq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::GreaterEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, less)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Less>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_bool)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, log)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a, vector<float>{expf(1), expf(2), expf(3), expf(4), expf(5), expf(6), expf(7), expf(8)});
    vector<float> loga;
    for (auto elt : read_vector<float>(a))
    {
        loga.push_back(logf(elt));
    }
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(loga, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f, 8.75f, -8.75f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, notequal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::NotEqual>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, select)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_broadcast)
{
    const string js =
        R"([{
       "name" : "Function_0",
       "ops" : [
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_4",
             "op" : "Parameter",
             "outputs" : ["Parameter_4"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_0",
             "op" : "Parameter",
             "outputs" : ["Parameter_0"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Constant_1",
             "op" : "Constant",
             "outputs" : ["Constant_1"],
             "shape" : [],
             "value" : ["0"]
           },
           {
             "axes" : [ 0, 1 ],
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : ["Constant_1"],
             "name" : "Broadcast_2",
             "op" : "Broadcast",
             "outputs" : ["Broadcast_2"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Parameter_0", "Broadcast_2" ],
             "name" : "Maximum_3",
             "op" : "Maximum",
             "outputs" : ["Maximum_3"]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Maximum_3", "Parameter_4" ],
             "name" : "Multiply_5",
             "op" : "Multiply",
             "outputs" : ["Multiply_5"]
           }
       ],
       "parameters" : [ "Parameter_0", "Parameter_4" ],
       "result" : ["Multiply_5"],
       "result_shape" : [ 3, 4 ],
       "result_type" :
           {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true}
    }])";
    stringstream ss(js);

    shared_ptr<Function> f = ngraph::deserialize(ss);

    // max(x,broadcast(Constant(0)))
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // If this compiles it works
}

NGRAPH_TEST(${BACKEND_NAME}, function_call)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto g =
        make_shared<Function>(make_shared<op::FunctionCall>(f, NodeVector{X + Y, Y + Z, Z + X}) +
                                  make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}),
                              op::ParameterVector{X, Y, Z});

    // Now call g on some test vectors.
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(g, {result}, {x, y, z});
    EXPECT_EQ((vector<float>{254, 368, 502, 656}), read_vector<float>(result));

    backend->call(g, {result}, {y, x, z});
    EXPECT_EQ((vector<float>{278, 400, 542, 704}), read_vector<float>(result));

    backend->call(g, {result}, {x, z, y});
    EXPECT_EQ((vector<float>{194, 296, 418, 560}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_to_non_existent_axis)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    ASSERT_THROW(auto f = make_shared<Function>(
                     make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}), op::ParameterVector{A}),
                 ngraph_error);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_trivial)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_colwise)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

// Test hybrid mechanism after broadcast
NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise_reversed)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto broadcast = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto reverse = make_shared<op::Reverse>(broadcast, AxisSet{1});
    auto f = make_shared<Function>(reverse, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise_int64)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i64, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_0)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_1)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 1, 2, 3, 4, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_2)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{2}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 2, 2, 3, 3, 4, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::f32), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_float32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

// Trivial case with no reduction axes.
NGRAPH_TEST(${BACKEND_NAME}, reduce_trivial)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, {});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_to_scalar)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_columns)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};

    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{66});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{66, 66, 66}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{66}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_cols_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{77});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{77, 77}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{77}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_vector_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{88});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{88}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{88}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_to_scalar_zero_by_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{99});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{99}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{99}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_3d_to_vector)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x*y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(make_shared<op::Multiply>(f_A, f_B), op::ParameterVector{f_A, f_B});

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(A, B, f, AxisSet{0, 1}),
                                   op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2v_012)
{
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_012)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_120)
{
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_s2t)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 1, 1, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{42});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_col)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_row)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2t_middle)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_same)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_transpose)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), read_vector<float>(result));
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
NGRAPH_TEST(${BACKEND_NAME}, reshape_6d)
{
    vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
    for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2, 2, 4, 3, 2};

    auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 237., 150., 222., 158., 230., 166., 238.,
            151., 223., 159., 231., 167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 257.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 244.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 282., 195., 267., 203., 275., 211., 283., 196., 268., 204., 276.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sin)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 6, -pi, pi};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, cos)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 3, -pi, pi};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return cosf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tan)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{pi / 4, 0.0f, -0.0f, 7 * pi / 4, 3 * pi / 4, 5 * pi / 4};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanf(x); });

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, asin)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return asinf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, acos)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return acosf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, atan)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return atanf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sinh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });

    backend->call(f, {result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, cosh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return coshf(x); });

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, tanh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanhf(x); });

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, exp)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-4, -3, -2, -1, 0, 1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto r = op::Constant::create(element::f32, Shape{}, {4.75});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {});
    EXPECT_EQ(vector<float>{4.75f}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto r = op::Constant::create(element::i64, Shape{}, {2112});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, Shape{});

    backend->call(f, {result}, {});
    EXPECT_EQ(vector<int64_t>{2112}, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::f32, shape, {4.75, 4.7, -5.3, 0.0});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {});
    EXPECT_EQ((vector<float>{4.75f, 4.7f, -5.3f, 0.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::i64, shape, {2112, 1848, 1776, 1964});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, shape);

    backend->call(f, {result}, {});
    EXPECT_EQ((vector<int64_t>{2112, 1848, 1776, 1964}), read_vector<int64_t>(result));
}

// Trivial case with no summed axes.
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19,
                             2 + 11 + 20,
                             3 + 12 + 21,
                             4 + 13 + 22,
                             5 + 14 + 23,
                             6 + 15 + 24,
                             7 + 16 + 25,
                             8 + 17 + 26,
                             9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 2 + 3,
                             4 + 5 + 6,
                             7 + 8 + 9,
                             10 + 11 + 12,
                             13 + 14 + 15,
                             16 + 17 + 18,
                             19 + 20 + 21,
                             22 + 23 + 24,
                             25 + 26 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                             2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                             3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                             8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar_stable)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1e-6f, -1, 0, 1});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(read_vector<float>(result), vector<float>{1e-6f}, 5e-2f));
    // EXPECT_EQ(vector<float>{1e-6}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_vector_stable)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 1,  1,  1,  1,  1,  1e-4f, 1e-5f, 1e-6f, 1,  1,  1,  1, 1,
                               1, -1, -1, -1, -1, -1, -1,    -1,    -1,    -1, -1, -1, -1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(read_vector<float>(result), vector<float>{1e-4f, 1e-5f, 1e-6f}, 5e-2f));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_5d_to_scalar)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2, 3, 4}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, std::vector<float>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ(std::vector<float>{243.}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_TRUE(test::all_close(vector<float>{1, 1, 729, 125}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_equality_bool)
{
    Shape shape{4};
    // auto A = make_shared<op::Parameter>(element::boolean, shape);
    // auto B = make_shared<op::Parameter>(element::boolean, shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {});
    EXPECT_EQ((vector<char>{true, false, true, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 2, 9, 10, 100, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{808});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{808}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{12};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(
        (vector<float>{0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 14, 15}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 1, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_fp_nonint_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.1f});
    auto result = backend->create_tensor(element::f32, shape_r);

    try
    {
        backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_oob_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{3000000});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 8};
    auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_far_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3000000, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 3, 3};
    auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_fp)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_fp_nonint)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1.01f, 0});
    auto result = backend->create_tensor(element::f32, shape_r);

    try
    {
        backend->call(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{921, 922, 925, 926, 937, 938, 941, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

                             16, 17, 18, 19, 20, 921, 922, 23, 24, 925, 926, 27, 28, 29, 30, 31,

                             32, 33, 34, 35, 36, 937, 938, 39, 40, 941, 942, 43, 44, 45, 46, 47,

                             48, 49, 50, 51, 52, 53,  54,  55, 56, 57,  58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 902, 908, 910, 932, 934, 940, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

                             16,  17, 18,  19, 20, 21, 22, 23, 24,  25, 26,  27, 28, 29, 30, 31,

                             932, 33, 934, 35, 36, 37, 38, 39, 940, 41, 942, 43, 44, 45, 46, 47,

                             48,  49, 50,  51, 52, 53, 54, 55, 56,  57, 58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 903, 908, 911, 932, 935, 940, 943});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// > from numpy import *
// > x = linspace(1,2*3*4,2*3*4)
// > y = linspace(1,3*4*5,3*4*5)
// > x.shape=(2,3,4)
// > y.shape=(3,4,5)
// > z = tensordot(x,y,([1,2],[0,1]))
// > z.shape = 2*5
// > z
// array([ 2938.,  3016.,  3094.,  3172.,  3250.,  7042.,  7264.,  7486.,
//         7708.,  7930.])
//
// Disabled because it doesn't work on CPU yet.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_3d_multi_axis)
{
    vector<float> a_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 5);
    for (int i = 0; i < 3 * 4 * 5; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 5};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 5};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2938., 3016., 3094., 3172., 3250., 7042., 7264., 7486., 7708., 7930.}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// > from numpy import *
// > x = array([6,61,2,3,5,21,75,23,23,0,23,2,35,67,1,2,9,16,2,3,6,1,8,0])
// > y = array([9,1,4,6,3,5,1,36,7,3,5,0,1,20,35,2,1,0,1,25,3,6,7,8])
// > x.shape=(2,4,3)
// > y.shape=(3,4,2)
// > z = tensordot(x,y,([2],[0]))
// > z.shape = 2*4*4*2
// > z
// array([ 483,  189,  331,   86,   85, 1262, 2155,  354,   83,   18,   58,
//         543,   77,  241,  325,  286,  859,  144,  438, 1025,  317,  973,
//        1041, 2930,  163,   69,  117,   50,   29,  472,  819,   62,  785,
//         236,  476,  235,  175, 1521, 2387, 1402,   97,   29,   69,  412,
//          63,  286,  429,  218,   45,   11,   29,  162,   27,  106,  149,
//         126,   65,   25,   44,    6,   11,  165,  281,   52])
//
// Disabled because it doesn't work on CPU yet.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_3d_one_axis_arbitrary)
{
    vector<float> a_data{6,  61, 2, 3, 5, 21, 75, 23, 23, 0, 23, 2,
                         35, 67, 1, 2, 9, 16, 2,  3,  6,  1, 8,  0};
    vector<float> b_data{9, 1,  4,  6, 3, 5, 1, 36, 7, 3, 5, 0,
                         1, 20, 35, 2, 1, 0, 1, 25, 3, 6, 7, 8};

    Shape shape_a{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 4, 4, 2};

    auto r = make_shared<op::Dot>(A, B);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{483,  189, 331, 86,  85,  1262, 2155, 354, 83,  18,   58,   543,  77,
                             241,  325, 286, 859, 144, 438,  1025, 317, 973, 1041, 2930, 163,  69,
                             117,  50,  29,  472, 819, 62,   785,  236, 476, 235,  175,  1521, 2387,
                             1402, 97,  29,  69,  412, 63,   286,  429, 218, 45,   11,   29,   162,
                             27,   106, 149, 126, 65,  25,   44,   6,   11,  165,  281,  52}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,3*4*2*3*2,3*4*2*2*3)
// x.shape=(2,3,3,4)
// y.shape=(3,4,2,2,3)
// z = tensordot(x,y,([2,3],[0,1]))
// z.shape = 2*3*2*2*3
// z
//
// array([  6942.,   7020.,   7098.,   7176.,   7254.,   7332.,   7410.,
//          7488.,   7566.,   7644.,   7722.,   7800.,  16590.,  16812.,
//         17034.,  17256.,  17478.,  17700.,  17922.,  18144.,  18366.,
//         18588.,  18810.,  19032.,  26238.,  26604.,  26970.,  27336.,
//         27702.,  28068.,  28434.,  28800.,  29166.,  29532.,  29898.,
//         30264.,  35886.,  36396.,  36906.,  37416.,  37926.,  38436.,
//         38946.,  39456.,  39966.,  40476.,  40986.,  41496.,  45534.,
//         46188.,  46842.,  47496.,  48150.,  48804.,  49458.,  50112.,
//         50766.,  51420.,  52074.,  52728.,  55182.,  55980.,  56778.,
//         57576.,  58374.,  59172.,  59970.,  60768.,  61566.,  62364.,
//         63162.,  63960.])
//
// Disabled because it doesn't work on CPU yet.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 2 * 2 * 3);
    for (int i = 0; i < 3 * 4 * 2 * 2 * 3; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 3, 2, 3, 2};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(
        (vector<float>{6942.,  7020.,  7098.,  7176.,  7254.,  7332.,  7410.,  7488.,  7566.,
                       7644.,  7722.,  7800.,  16590., 16812., 17034., 17256., 17478., 17700.,
                       17922., 18144., 18366., 18588., 18810., 19032., 26238., 26604., 26970.,
                       27336., 27702., 28068., 28434., 28800., 29166., 29532., 29898., 30264.,
                       35886., 36396., 36906., 37416., 37926., 38436., 38946., 39456., 39966.,
                       40476., 40986., 41496., 45534., 46188., 46842., 47496., 48150., 48804.,
                       49458., 50112., 50766., 51420., 52074., 52728., 55182., 55980., 56778.,
                       57576., 58374., 59172., 59970., 60768., 61566., 62364., 63162., 63960.}),
        read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,2*3*3*4*2,2*3*3*4*2)
// x.shape=(2,3,3,4)
// y.shape=(2,3,3,4,2)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// z
//
// array([ 251412.,  254040.])
//
// Disabled because it doesn't work on CPU yet.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_more)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 4 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 4 * 2; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{251412., 254040.}), read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,20*30*30*40,20*30*30*40)
// y = linspace(1,20*30*30*40*20,20*30*30*40*20)
// x.shape=(20,30,30,40)
// y.shape=(20,30,30,40,20)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// set_printoptions(precision=20)
// z
//
// array([  2.48832025919525478400e+18,   2.48832051839533977600e+18,
//          2.48832077759658444800e+18,   2.48832103679413504000e+18,
//          2.48832129599669350400e+18,   2.48832155519793971200e+18,
//          2.48832181439802265600e+18,   2.48832207359808000000e+18,
//          2.48832233279813580800e+18,   2.48832259199822028800e+18,
//          2.48832285119946496000e+18,   2.48832311040043008000e+18,
//          2.48832336959957401600e+18,   2.48832362880081817600e+18,
//          2.48832388800090368000e+18,   2.48832414720096000000e+18,
//          2.48832440640101478400e+18,   2.48832466560109772800e+18,
//          2.48832492480234188800e+18,   2.48832518400031897600e+18])
//
// Disabled because this test is very slow.
//
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_big_fp64_VERY_SLOW)
{
    vector<double> a_data(20 * 30 * 30 * 40);
    for (int i = 0; i < 20 * 30 * 30 * 40; i++)
    {
        a_data[i] = double(i + 1);
    }

    vector<double> b_data(20 * 30 * 30 * 40 * 20);
    for (int i = 0; i < 20 * 30 * 30 * 40 * 20; i++)
    {
        b_data[i] = double(i + 1);
    }

    Shape shape_a{20, 30, 30, 40};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    Shape shape_b{20, 30, 30, 40, 20};
    auto B = make_shared<op::Parameter>(element::f64, shape_b);
    Shape shape_r{20};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f64, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f64, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_TRUE(test::all_close(
        vector<double>{
            2.48832025919525478400e+18, 2.48832051839533977600e+18, 2.48832077759658444800e+18,
            2.48832103679413504000e+18, 2.48832129599669350400e+18, 2.48832155519793971200e+18,
            2.48832181439802265600e+18, 2.48832207359808000000e+18, 2.48832233279813580800e+18,
            2.48832259199822028800e+18, 2.48832285119946496000e+18, 2.48832311040043008000e+18,
            2.48832336959957401600e+18, 2.48832362880081817600e+18, 2.48832388800090368000e+18,
            2.48832414720096000000e+18, 2.48832440640101478400e+18, 2.48832466560109772800e+18,
            2.48832492480234188800e+18, 2.48832518400031897600e+18},
        read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image)
{
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image)
{
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image)
{
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_overpadded)
{
    Shape shape_a{1, 1, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 7, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    auto min = std::numeric_limits<float>::lowest();
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{min, min, min, min, min},
                                                           {1, 2, 2, 2, 1},
                                                           {3, 3, 2, 2, 1},
                                                           {3, 3, 2, 1, 1},
                                                           {2, 1, 2, 2, 2},
                                                           {2, 2, 2, 2, 2},
                                                           {2, 2, 1, 0, 0}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_padded)
{
    Shape shape_a{1, 1, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 6, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{1, 2, 2, 2, 1},
                                          {3, 3, 2, 2, 1},
                                          {3, 3, 2, 1, 1},
                                          {2, 1, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 1, 0, 0}}}})
                   .get_vector()),
              read_vector<float>(result));
}

// Test to make sure that negative elements and padding are handled properly. Added this because
// mkldnn calls its padding "zero padding" but apparently that is not technically true (negative
// values still "win" versus out-of-bounds values), which is good.
NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_padded_negative_values)
{
    auto shape_a = Shape{
        1,
        1,
        1,
        14}; // 1 image, 1 channel, 1 row, 14 columns (if it's 1D we don't get mkldnn as of this writing)
    Shape window_shape{1, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 1};
    Shape padding_above{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 15};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>{{{{-1, -2, -3, -3, -2, -1, -3, -2, -2, -2, -2, -3, -4, -5}}}}
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 4>({{{{-1, -1, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2, -3, -4, -5}}}})
             .get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_strided)
{
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(A, window_shape, window_movement_strides), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, not)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 2, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_0d)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_nochange)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 1, 2, 3, 4, 5, 6, 7}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_0)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{7, 6, 5, 4, 3, 2, 1, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_nochange)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_0)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_nochange)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                        {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_0)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
                                        {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_1)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
                                        {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_2)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
                                        {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_01)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
                                        {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_02)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
                                        {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_12)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1, 2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
                                        {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_012)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
                                        {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, abc_tbb)
{
    // Force TBB flow graph generation in the CPU backend
    // This has no effect on other backends
    bool use_tbb = (getenv("NGRAPH_CPU_USE_TBB") != nullptr);
    if (!use_tbb)
    {
        setenv("NGRAPH_CPU_USE_TBB", "1", 1);
    }

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());

    if (!use_tbb)
    {
        unsetenv("NGRAPH_CPU_USE_TBB");
    }
}

//
// The unit tests for ReduceWindow follow exactly what we test for MaxPool---but they use ReduceWindow to do it.
//
NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_1image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 5, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 4, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_1channel_1image_strided)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 8, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 3, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_with_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 5};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 2};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}, {1, 5, 7, 5, 6}, {0, 6, 2, 10, 2}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 0}, {0, 0, 8, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, 0, 1, 0}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 6};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 1}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// Adapted from the XLA docs to provide an example in >2D: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_3d_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{2, 4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 4, 6};
    Shape window_shape{2, 2, 3};
    auto window_strides = Strides{2, 2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        test::NDArray<float, 3>(
            {{{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}},
             {{2, 5, 8, 3, 4, 2}, {1, 2, 8, 4, 5, 2}, {10, 2, 3, 4, 1, 0}, {4, 1, 2, 4, 5, 7}}})
            .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 3>({{{2, 6}, {3, 1}}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(
        (test::NDArray<float, 3>(
             {{{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
              {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}})
             .get_vector()),
        read_vector<float>(result));
}

template <typename OP>
void make_unary_empty_test(const string& backend_name)
{
    Shape shape{0};

    op::ParameterVector params;
    NodeVector result_list;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(s_known_element_types[i], shape);
        params.push_back(p);
        result_list.push_back(make_shared<OP>(p));
    }

    auto f = make_shared<Function>(result_list, params);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::TensorView>> inputs;
    vector<shared_ptr<runtime::TensorView>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
    }

    backend->call(f, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
}

template <typename OP>
void make_binary_empty_test(const string& backend_name, bool is_comparison = false)
{
    Shape shape{0};
    op::ParameterVector A;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        A.push_back(make_shared<op::Parameter>(s_known_element_types[i], shape));
    }

    NodeVector result_list;
    for (shared_ptr<op::Parameter> p : A)
    {
        result_list.push_back(make_shared<OP>(p, p));
    }

    auto f = make_shared<Function>(result_list, A);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::TensorView>> inputs;
    vector<shared_ptr<runtime::TensorView>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        if (is_comparison)
        {
            outputs.push_back(backend->create_tensor(element::from<char>(), shape));
        }
        else
        {
            outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        }
    }

    backend->call(f, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    if (is_comparison)
    {
        EXPECT_EQ(read_vector<char>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[9]).size(), 0);
    }
    else
    {
        EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_abs)
{
    make_unary_empty_test<op::Abs>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_ceiling)
{
    make_unary_empty_test<op::Ceiling>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_exp)
{
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_floor)
{
    make_unary_empty_test<op::Floor>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_log)
{
    make_unary_empty_test<op::Log>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_negative)
{
    make_unary_empty_test<op::Negative>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not)
{
    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::from<char>(), shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::from<char>(), shape);
    auto result = backend->create_tensor(element::from<char>(), shape);

    backend->call(f, {result}, {a});

    auto in_vec = read_vector<char>(a);
    auto out_vec = read_vector<char>(result);

    EXPECT_EQ(in_vec.size(), 0);
    EXPECT_EQ(out_vec.size(), 0);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sign)
{
    make_unary_empty_test<op::Sign>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sqrt)
{
    make_unary_empty_test<op::Sqrt>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sin)
{
    make_unary_empty_test<op::Sin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sinh)
{
    make_unary_empty_test<op::Sinh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cos)
{
    make_unary_empty_test<op::Cos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cosh)
{
    make_unary_empty_test<op::Cosh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tan)
{
    make_unary_empty_test<op::Tan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tanh)
{
    make_unary_empty_test<op::Tanh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_asin)
{
    make_unary_empty_test<op::Asin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_acos)
{
    make_unary_empty_test<op::Acos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_atan)
{
    make_unary_empty_test<op::Atan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_add)
{
    make_binary_empty_test<op::Add>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_divide)
{
    make_binary_empty_test<op::Divide>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_eq)
{
    make_binary_empty_test<op::Equal>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greater)
{
    make_binary_empty_test<op::Greater>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greatereq)
{
    make_binary_empty_test<op::GreaterEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_less)
{
    make_binary_empty_test<op::Less>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_lesseq)
{
    make_binary_empty_test<op::LessEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_maximum)
{
    make_binary_empty_test<op::Maximum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_minimum)
{
    make_binary_empty_test<op::Minimum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_multiply)
{
    make_binary_empty_test<op::Multiply>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not_equal)
{
    make_binary_empty_test<op::NotEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_power)
{
    make_binary_empty_test<op::Power>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_subtract)
{
    make_binary_empty_test<op::Subtract>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_outlining)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto conv2 = make_shared<op::Convolution>(conv1,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto f = make_shared<Function>(conv2, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(vector<float>{expected_result}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mkldnn_layouts)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool1 = make_shared<op::AvgPool>(conv1, pool_shape);
    auto pool1_result = make_shared<op::Result>(pool1);
    // Request result in default layout
    pool1_result->set_needs_default_layout(true);
    auto f = make_shared<Function>(ResultVector{pool1_result}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> input(64, 1.0f);
    vector<float> weights;
    vector<float> rv(128);
    for (int i = 0; i < 128; i++)
        weights.push_back(0.0f);
    for (int i = 0; i < 384; i++)
        weights.push_back(1.0f);

    auto a = backend->create_tensor(element::f32, shape_a, input.data());
    auto b = backend->create_tensor(element::f32, shape_b, weights.data());
    auto result = backend->create_tensor(element::f32, shape_r, rv.data());

    vector<float> expected_result;
    for (int i = 0; i < 32; i++)
        expected_result.push_back(0.0f);
    for (int i = 0; i < 96; i++)
        expected_result.push_back(16.0f);

    backend->call(f, {result}, {a, b});

    EXPECT_EQ(vector<float>{expected_result}, rv);
}

NGRAPH_TEST(${BACKEND_NAME}, computation_reuse)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv = make_shared<op::Convolution>(A,
                                             B,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool = make_shared<op::AvgPool>(conv, pool_shape);
    auto bias = make_shared<op::Broadcast>(
        op::Constant::create(element::f32, Shape{}, {2.14}), shape_r, AxisSet{0, 1, 2, 3});
    auto result_op = make_shared<op::Result>(pool + bias);
    auto f = make_shared<Function>(ResultVector{result_op}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> input(64, 1.0f);
    vector<float> weights(512, 0.5f);
    vector<float> rv(128);

    auto a = backend->create_tensor(element::f32, shape_a, input.data());
    auto b = backend->create_tensor(element::f32, shape_b, weights.data());
    auto result = backend->create_tensor(element::f32, shape_r, rv.data());

    backend->call(f, {result}, {a, b});

    vector<float> rv_saved(rv);

    b->set_stale(false);
    backend->call(f, {result}, {a, b});
    EXPECT_EQ(rv_saved, rv);
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image)
{
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 3>({{{1 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           4 / denom,
                                                           5 / denom,
                                                           5 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           0 / denom}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image)
{
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 3>({{{1 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           4 / denom,
                                                           5 / denom,
                                                           5 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           0 / denom}},
                                                         {{3 / denom,
                                                           4 / denom,
                                                           2 / denom,
                                                           1 / denom,
                                                           0 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           3 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           3 / denom}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_2channel_2image)
{
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 3>({{{1 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           4 / denom,
                                                           5 / denom,
                                                           5 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           0 / denom},
                                                          {0 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           5 / denom,
                                                           5 / denom,
                                                           4 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           3 / denom,
                                                           1 / denom}},

                                                         {{3 / denom,
                                                           4 / denom,
                                                           2 / denom,
                                                           1 / denom,
                                                           0 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           3 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           3 / denom},
                                                          {3 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           1 / denom,
                                                           3 / denom,
                                                           2 / denom,
                                                           2 / denom,
                                                           0 / denom,
                                                           1 / denom,
                                                           2 / denom,
                                                           4 / denom,
                                                           3 / denom}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call(f, {result}, {a});

    EXPECT_TRUE(test::all_close(
        test::NDArray<float, 4>({{{{6 / denom, 8 / denom, 5 / denom}, // img 0 chan 0
                                   {7 / denom, 5 / denom, 3 / denom},
                                   {5 / denom, 2 / denom, 5 / denom},
                                   {6 / denom, 5 / denom, 5 / denom}},

                                  {{5 / denom, 7 / denom, 6 / denom}, // img 0 chan 1
                                   {8 / denom, 6 / denom, 7 / denom},
                                   {7 / denom, 2 / denom, 3 / denom},
                                   {6 / denom, 1 / denom, 0 / denom}}},

                                 {{{5 / denom, 6 / denom, 5 / denom}, // img 1 chan 0
                                   {3 / denom, 5 / denom, 9 / denom},
                                   {3 / denom, 6 / denom, 9 / denom},
                                   {2 / denom, 3 / denom, 3 / denom}},

                                  {{5 / denom, 3 / denom, 1 / denom}, // img 1 chan 1
                                   {6 / denom, 5 / denom, 4 / denom},
                                   {7 / denom, 5 / denom, 6 / denom},
                                   {4 / denom, 2 / denom, 4 / denom}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_strided)
{
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(A, window_shape, window_movement_strides), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                                           {6 / denom, 5 / denom, 8 / denom},
                                                           {6 / denom, 2 / denom, 4 / denom}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded)
{
    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                  {{3.0f / 1, 8.0f / 2, 7.0f / 2, 2.0f / 1},
                                                   {5.0f / 2, 10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                   {5.0f / 2, 11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                   {3.0f / 1, 9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_below)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2},
                                                           {0.0f / 2, 4.0f / 4, 6.0f / 4},
                                                           {2.0f / 2, 5.0f / 4, 5.0f / 4}},
                                                          {{3.0f / 1, 8.0f / 2, 7.0f / 2},
                                                           {5.0f / 2, 10.0f / 4, 16.0f / 4},
                                                           {5.0f / 2, 11.0f / 4, 20.0f / 4}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_above)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                           {5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                           {2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                          {{10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                           {11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                           {9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(
        test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 2, 0.0f / 1},
                                   {0.0f / 2, 4.0f / 4, 6.0f / 6, 6.0f / 4, 2.0f / 2},
                                   {2.0f / 3, 6.0f / 6, 8.0f / 9, 6.0f / 6, 2.0f / 3},
                                   {2.0f / 2, 5.0f / 4, 7.0f / 6, 5.0f / 4, 2.0f / 2},
                                   {2.0f / 1, 2.0f / 2, 2.0f / 3, 0.0f / 2, 0.0f / 1}},
                                  {{3.0f / 1, 8.0f / 2, 10.0f / 3, 7.0f / 2, 2.0f / 1},
                                   {5.0f / 2, 10.0f / 4, 21.0f / 6, 16.0f / 4, 11.0f / 2},
                                   {8.0f / 3, 19.0f / 6, 35.0f / 9, 27.0f / 6, 16.0f / 3},
                                   {5.0f / 2, 11.0f / 4, 25.0f / 6, 20.0f / 4, 14.0f / 2},
                                   {3.0f / 1, 9.0f / 2, 14.0f / 3, 11.0f / 2, 5.0f / 1}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 3, 0.0f / 1},
                                                           {2.0f / 3, 8.0f / 9, 2.0f / 3},
                                                           {2.0f / 1, 2.0f / 3, 0.0f / 1}},
                                                          {{3.0f / 1, 10.0f / 3, 2.0f / 1},
                                                           {8.0f / 3, 35.0f / 9, 16.0f / 3},
                                                           {3.0f / 1, 14.0f / 3, 5.0f / 1}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided_uneven)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(
        test::NDArray<float, 4>(
            {{{{0.0f / 1, 1.0f / 2}, {2.0f / 3, 6.0f / 6}, {2.0f / 1, 0.0f / 2}},
              {{3.0f / 1, 7.0f / 2}, {8.0f / 3, 27.0f / 6}, {3.0f / 1, 11.0f / 2}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {1, 2112, 2112, 2, 2112, 2112, 3, 2112, 2112, 4, 2112, 2112, 5, 2112, 2112, 6})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{15};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{25};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1,    2112, 2112, 2, 2112,
                                        2112, 3,    2112, 2112, 4,    2112, 2112, 5, 2112,
                                        2112, 6,    2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_2d)
{
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{7, 6};
    Shape padding_below{1, 0};
    Shape padding_above{2, 1};
    Shape padding_interior{2, 1};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{9});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{9, 9, 9, 9, 9, 9},
                                        {1, 9, 2, 9, 3, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {4, 9, 5, 9, 6, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 3};
    Shape padding_above{3, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({{}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    Shape shape_a{0, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 1};
    Shape padding_above{3, 1};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{1, 3};
    Shape padding_above{1, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 4, 4};
    Shape padding_below{0, 0, 1, 1};
    Shape padding_above{0, 0, 1, 1};
    Shape padding_interior{0, 0, 0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call(f, {result}, {a, b});
    // clang-format off
    EXPECT_EQ((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result));
    // clang-format on
}

// This is a regression test for one of TF's unit tests, which was failing.
// The problem was inappropriate handling of the shape computation for a
// zero-length axis with interior padding. Rather than subtract 1 from the
// source shape and multiply by the interior padding (which causes underflow),
// we should just count the pre-interior-padding length as zero.
NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_4d_2x0x3x2)
{
    Shape shape_a{2, 0, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape padding_below{1, 0, 0, 0};
    Shape padding_above{0, 2, 0, 0};
    Shape padding_interior{2, 1, 0, 0};
    Shape shape_r{5, 2, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected(5 * 2 * 3 * 2, 2112);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(expected, read_vector<float>(result));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, product_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, product_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{24}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{15, 48}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 12, 30}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 * 10 * 19,
                             2 * 11 * 20,
                             3 * 12 * 21,
                             4 * 13 * 22,
                             5 * 14 * 23,
                             6 * 15 * 24,
                             7 * 16 * 25,
                             8 * 17 * 26,
                             9 * 18 * 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1 * 2 * 3,
                             4 * 5 * 6,
                             7 * 8 * 9,
                             10 * 11 * 12,
                             13 * 14 * 15,
                             16 * 17 * 18,
                             19 * 20 * 21,
                             22 * 23 * 24,
                             25 * 26 * 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f * 12.0f * 3.0f *
                             2.0f * 11.0f * 8.0f * 5.0f * 14.0f * 5.0f * 8.0f * 11.0f * 2.0f *
                             3.0f * 12.0f * 7.0f * 6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1}), read_vector<float>(result));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, max_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, max_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{4}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{5, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{25.0f, 26.0f, 27.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{14.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float mi = -std::numeric_limits<float>::infinity();

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{mi, mi, mi, mi, mi, mi}), read_vector<float>(result));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, min_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, min_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 10, 13, 16, 19, 22, 25}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float inf = std::numeric_limits<float>::infinity();

    backend->call(f, {result}, {a});
    EXPECT_EQ((vector<float>{inf, inf, inf, inf, inf, inf}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    backend->call(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    backend->call(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, op::ParameterVector{B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    backend->call(f, {result}, {b});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dbackprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};

    backend->call(f, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dbackprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    backend->call(f, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-3, -2, -1, 0, 1, 2});
    auto result = backend->create_tensor(element::f32, shape);

    auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);

    backend->call(f, {result}, {a});
    vector<float> expected{
        expf(-3) / d, expf(-2) / d, expf(-1) / d, expf(0) / d, expf(1) / d, expf(2) / d};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));

    // empty AxisSet is the same as "full" AxisSet
    f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{}), op::ParameterVector{A});
    backend = runtime::Backend::create("${BACKEND_NAME}");

    backend->call(f, {result}, {a});
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-20) + expf(-30);
    auto d1 = expf(-40) + expf(-50) + expf(-60);

    backend->call(f, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d0,
                           expf(-30) / d0,
                           expf(-40) / d1,
                           expf(-50) / d1,
                           expf(-60) / d1};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_underflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto low = std::numeric_limits<float>::lowest();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{low, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(low) + expf(3);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    backend->call(f, {result}, {a});
    vector<float> expected{
        expf(low) / d0, expf(1) / d1, expf(2) / d2, expf(3) / d0, expf(4) / d1, expf(5) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, multiple_backends)
{
    Shape shape{2, 2};
    auto A1 = make_shared<op::Parameter>(element::f32, shape);
    auto B1 = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A1 + B1, op::ParameterVector{A1, B1});

    auto A2 = make_shared<op::Parameter>(element::f32, shape);
    auto B2 = make_shared<op::Parameter>(element::f32, shape);
    auto g = make_shared<Function>(A2 * B2, op::ParameterVector{A2, B2});

    auto backend1 = runtime::Backend::create("${BACKEND_NAME}");

    auto backend2 = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result1 = backend1->create_tensor(element::f32, shape);

    shared_ptr<runtime::TensorView> a2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result2 = backend2->create_tensor(element::f32, shape);

    copy_data(a1, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b1, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    copy_data(a2, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b2, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend1->call(f, {result1}, {a1, b1});
    EXPECT_EQ(read_vector<float>(result1),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());

    backend2->call(g, {result2}, {a2, b2});
    EXPECT_EQ(read_vector<float>(result2),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, tensorview_custom_mem)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

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
    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), rv);
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);
    auto d = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call(f, {a}, {c, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_and)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::And>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 0, 0, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_or)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Or>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 1, 1, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_fprop_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   op::ParameterVector{input, gamma, beta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    vector<float> expected_mean{0.602912f, 0.599727f};
    vector<float> expected_variance{0.00472505f, 0.0361782f};

    backend->call(f, {bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean), 1e-5f, 1e-6f));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_fprop_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   op::ParameterVector{input, gamma, beta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    backend->call(f, {bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_bprop_n4c3h2w2)
{
    auto input_shape = Shape{4, 3, 2, 2};
    auto shape_mean = Shape{3};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{3};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{3};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{3};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{3};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{4, 3, 2, 2};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input);
    auto bn_dx = make_shared<op::GetOutputElement>(bn, 0);
    auto bn_dgamma = make_shared<op::GetOutputElement>(bn, 1);
    auto bn_dbeta = make_shared<op::GetOutputElement>(bn, 2);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto _input = backend->create_tensor(element::f32, input_shape);
    vector<float> dataInput{
        10.76331902f, 11.51178265f, 10.31018162f, 12.2993021f,  14.17626667f, 14.63498497f,
        13.63494492f, 13.84248161f, 11.34602547f, 13.22014618f, 10.46686649f, 10.39842987f,
        12.94806862f, 11.71670246f, 14.94438076f, 13.13236618f, 13.40889645f, 12.76128387f,
        11.34430027f, 11.86629677f, 11.11464024f, 10.93221283f, 11.95324039f, 10.96581173f,
        13.05455494f, 14.41404247f, 13.11169434f, 11.26559448f, 10.89965153f, 14.08202171f,
        11.12685776f, 12.58428574f, 12.59247875f, 13.00187492f, 12.66310215f, 10.06655025f,
        12.62048626f, 14.47942352f, 13.84950638f, 10.61425877f, 11.47936344f, 13.06011772f,
        13.63069057f, 12.31748772f, 13.84555244f, 10.95815468f, 12.78933334f, 12.75389099f};
    copy_data(_input, dataInput);
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{12.56472874f, 12.80312157f, 11.81676865f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{1.94557643f, 1.32772446f, 1.28163588f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{2.0f, 2.0f, 2.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    shared_ptr<runtime::TensorView> _delta = backend->create_tensor(element::f32, shape_r);
    vector<float> deltaData(shape_size(shape_r), 20.0f);
    copy_data(_delta, deltaData);

    auto f = make_shared<Function>(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                   op::ParameterVector{mean, var, input, gamma, beta});

    auto C = std::make_shared<op::Parameter>(element::f32, shape_r);

    auto zero = ngraph::make_zero(bn_dgamma->get_element_type(), bn_dgamma->get_shape());
    ngraph::autodiff::Adjoints adjoints(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                        NodeVector{C, zero, zero});

    auto dinput = adjoints.backprop_node(input);
    auto dgamma = adjoints.backprop_node(gamma);
    auto dbeta = adjoints.backprop_node(beta);

    auto df = make_shared<Function>(NodeVector{dinput, dgamma, dbeta},
                                    op::ParameterVector{mean, var, input, gamma, beta, C});

    //roundtrip serialization
    string js = serialize(df, 4);
    istringstream in(js);
    df = deserialize(in);

    shared_ptr<runtime::TensorView> _dinput = backend->create_tensor(element::f32, shape_r);
    shared_ptr<runtime::TensorView> _dgamma = backend->create_tensor(element::f32, gamma_shape);
    shared_ptr<runtime::TensorView> _dbeta = backend->create_tensor(element::f32, beta_shape);

    backend->call(df, {_dinput, _dgamma, _dbeta}, {_mean, _var, _input, _gamma, _beta, _delta});

    vector<float> expected_input{
        8.17051607e-06f,  4.77576657e-06f,  1.02257760e-05f,  1.20387525e-06f,  -1.73868522e-06f,
        3.84632768e-06f,  -1.07932050e-05f, -2.57458956e-06f, -2.22166714e-06f, -8.38779043e-06f,
        -2.48082982e-06f, 5.89238360e-06f,  -2.52895109e-07f, -8.68433445e-06f, -5.82726737e-06f,
        8.84659658e-06f,  3.03944108e-05f,  4.05480879e-05f,  1.84123158e-05f,  2.30061178e-05f,
        1.34087590e-05f,  -9.26072571e-07f, -3.22908454e-05f, -2.07365116e-05f, -4.21330941e-05f,
        2.83083100e-05f,  -3.71039101e-05f, -4.84390640e-06f, -2.93012376e-05f, 5.68858087e-06f,
        1.83181458e-05f,  -1.07494506e-05f, -2.32429103e-06f, 6.92914809e-06f,  -6.66512321e-06f,
        -7.00302840e-06f, -3.46675184e-06f, -4.36748381e-06f, 6.73822226e-07f,  -4.20158993e-06f,
        3.83005061e-06f,  5.85143729e-06f,  4.17875243e-06f,  -8.64167783e-06f, 1.00170803e-05f,
        -4.23939666e-06f, 4.80201680e-06f,  4.62702078e-06f};

    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dinput), expected_input, 1e-3f, 1e-4f));
    vector<float> expected_dgamma{7.06315041e-05f, -2.35289335e-04f, -5.06639481e-05f};
    ASSERT_TRUE(
        ngraph::test::all_close(read_vector<float>(_dgamma), expected_dgamma, 1e-2f, 1e-3f));
    vector<float> expected_dbeta{320.f, 320.f, 320.f};
    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dbeta), expected_dbeta, 1e-4f, 1e-8f));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_fprop_inference_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var);

    auto f = make_shared<Function>(bn, op::ParameterVector{input, gamma, beta, mean, var});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);
    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    backend->call(f, {bn_output}, {_input, _gamma, _beta, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_fprop_globalstats_b2c2w2h1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var, true);

    auto f = make_shared<Function>(bn, op::ParameterVector{gamma, beta, input, mean, var});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);
    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    backend->call(f, {bn_output}, {_gamma, _beta, _input, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}
