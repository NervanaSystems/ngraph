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

#include <gtest/gtest.h>

#include "ngraph/op/add.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(async, execute)
{
    Shape shape{100000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("INTERPRETER");

    vector<float> data(shape_size(shape), 2);
    vector<float> result_data(shape_size(shape), 0);

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape, data.data());
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape, data.data());
    shared_ptr<runtime::Tensor> r = backend->create_tensor(element::f32, shape, result_data.data());

    auto handle = backend->compile(f);
    auto future = handle->begin_execute({r}, {a, b});
    ASSERT_TRUE(future.valid());
    future.get();

    for (float x : result_data)
    {
        ASSERT_EQ(x, 4);
    }
}

TEST(async, tensor_read_write)
{
    chrono::milliseconds ten_ms(100);
    Shape shape{100000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("INTERPRETER");
    auto handle = backend->compile(f);

    vector<float> data(shape_size(shape), 2);
    vector<float> data_r(shape_size(shape), 0);

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> r = backend->create_tensor(element::f32, shape);

    auto future_a = a->begin_write(data.data(), data.size() * sizeof(float), 0);
    auto future_b = b->begin_write(data.data(), data.size() * sizeof(float), 0);
    ASSERT_TRUE(future_a.valid());
    ASSERT_TRUE(future_b.valid());

    auto future = handle->begin_execute({r}, {a, b});

    // get() waits for the result to be ready
    future.get();

    auto future_r = r->begin_read(data_r.data(), data_r.size() * sizeof(float), 0);
    ASSERT_TRUE(future_r.valid());

    EXPECT_EQ(future_a.wait_for(ten_ms), future_status::ready);
    EXPECT_EQ(future_b.wait_for(ten_ms), future_status::ready);
    EXPECT_EQ(future_r.wait_for(ten_ms), future_status::ready);

    for (float x : data_r)
    {
        ASSERT_EQ(x, 4);
    }
}
