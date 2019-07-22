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

// End to end test for MLIR. Add tests here that are specific to test MLIR functionality
// MLIR is implicitly tested during other unit-tests as well.

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

// Combined ops test
NGRAPH_TEST(${BACKEND_NAME}, mlir_dot_add)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto dot = make_shared<op::Dot>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape_in1);
    auto add = make_shared<op::Add>(dot, C);
    auto f = make_shared<Function>(add, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    copy_data(c, vector<float>{5.f, 4.f, 3.f, 2.f, 1.f, 0.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{35.f, 40.f, 45.f, 68.f, 82.f, 96.f}));
}

// Sub-graph extraction tests
NGRAPH_TEST(${BACKEND_NAME}, mlir_subgraphs_dot_add)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};

    // sub-graph 1
    auto P1 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P2 = make_shared<op::Parameter>(element::f32, shape_in2);
    auto P3 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto dot = make_shared<op::Dot>(P1, P2);
    auto sg1_output = make_shared<op::Add>(dot, P3);

    // sub-graph 2
    auto P4 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P5 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P6 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto add = make_shared<op::Add>(P4, P5);
    auto sg2_output = make_shared<op::Add>(add, P6);

    auto out = make_shared<op::Maximum>(sg1_output, sg2_output);

    auto f = make_shared<Function>(out, ParameterVector{P1, P2, P3, P4, P5, P6});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> p1 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p2 = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> p3 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p4 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p5 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p6 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(p1, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p2, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    copy_data(p3, vector<float>{5.f, 4.f, 3.f, 2.f, 1.f, 0.f});

    copy_data(p4, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p5, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p6, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {p1, p2, p3, p4, p5, p6});
    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{35, 40, 45, 68, 82, 96}));
}

NGRAPH_TEST(${BACKEND_NAME}, mlir_subgraphs_dot_add_2)
{
    // Tests 2 sub-graphs merged at a join point into one.
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};

    // sub-graph 1
    auto P1 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P2 = make_shared<op::Parameter>(element::f32, shape_in2);
    auto P3 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto dot = make_shared<op::Dot>(P1, P2);
    auto sg1_output = make_shared<op::Add>(dot, P3);

    // sub-graph 2
    auto P4 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P5 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P6 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto add = make_shared<op::Add>(P4, P5);
    auto sg2_output = make_shared<op::Add>(add, P6);

    auto add2 = make_shared<op::Add>(sg1_output, sg2_output);
    auto abs = make_shared<op::Abs>(add2);

    auto f = make_shared<Function>(abs, ParameterVector{P1, P2, P3, P4, P5, P6});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> p1 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p2 = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> p3 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p4 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p5 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p6 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(p1, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p2, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    copy_data(p3, vector<float>{5.f, 4.f, 3.f, 2.f, 1.f, 0.f});

    copy_data(p4, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p5, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p6, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {p1, p2, p3, p4, p5, p6});
    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{38, 46, 54, 80, 97, 114}));
}

NGRAPH_TEST(${BACKEND_NAME}, mlir_subgraphs_dot_add_3)
{
    // Tests 3 distinct sub-graphs
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};

    // sub-graph 1
    auto P1 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P2 = make_shared<op::Parameter>(element::f32, shape_in2);
    auto P3 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto dot = make_shared<op::Dot>(P1, P2);
    auto sg1_output = make_shared<op::Add>(dot, P3);

    // sub-graph 2
    auto P4 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P5 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P6 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto add = make_shared<op::Add>(P4, P5);
    auto sg2_output = make_shared<op::Add>(add, P6);

    auto max = make_shared<op::Maximum>(sg1_output, sg2_output);
    auto add2 = make_shared<op::Add>(max, max);

    auto f = make_shared<Function>(add2, ParameterVector{P1, P2, P3, P4, P5, P6});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> p1 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p2 = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> p3 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p4 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p5 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p6 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(p1, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p2, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    copy_data(p3, vector<float>{5.f, 4.f, 3.f, 2.f, 1.f, 0.f});

    copy_data(p4, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p5, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p6, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {p1, p2, p3, p4, p5, p6});
    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{70, 80, 90, 136, 164, 192}));
}

NGRAPH_TEST(${BACKEND_NAME}, mlir_subgraphs_cycle)
{
    // Tests 3 distinct sub-graphs
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};

    // sub-graph 1
    auto P1 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto P2 = make_shared<op::Parameter>(element::f32, shape_in2);
    auto P3 = make_shared<op::Parameter>(element::f32, shape_in1);
    auto dot = make_shared<op::Dot>(P1, P2);
    auto add = make_shared<op::Add>(dot, P3);
    auto abs = make_shared<op::Abs>(add);
    auto add2 = make_shared<op::Add>(add, abs);

    auto f = make_shared<Function>(add2, ParameterVector{P1, P2, P3});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> p1 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> p2 = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> p3 = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(p1, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(p2, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    copy_data(p3, vector<float>{5.f, 4.f, 3.f, 2.f, 1.f, 0.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {p1, p2, p3});
    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{70, 80, 90, 136, 164, 192}));
}
