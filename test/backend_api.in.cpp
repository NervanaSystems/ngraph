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
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// This tests a backend's implementation of the two parameter version of create_tensor
NGRAPH_TEST(${BACKEND_NAME}, create_tensor_1)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> av = {1, 2, 3, 4};
    vector<float> bv = {5, 6, 7, 8};
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, av);
    copy_data(b, bv);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected = {6, 8, 10, 12};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

// This tests a backend's implementation of the three parameter version of create_tensor
NGRAPH_TEST(${BACKEND_NAME}, create_tensor_2)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> av = {1, 2, 3, 4};
    vector<float> bv = {5, 6, 7, 8};
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape, av.data());
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape, bv.data());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected = {6, 8, 10, 12};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

// This tests a backend's implementation of the copy_from for tensor
NGRAPH_TEST(${BACKEND_NAME}, tensor_copy_from)
{
    Shape shape{2, 2};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> av = {1, 2, 3, 4};
    vector<float> bv = {5, 6, 7, 8};
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    copy_data(a, av);
    copy_data(b, bv);

    a->copy_from(*b);
    EXPECT_TRUE(test::all_close(bv, read_vector<float>(a), 0.0f, 0.0f));
}
