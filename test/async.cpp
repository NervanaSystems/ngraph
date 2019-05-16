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
    bool rc = future.get();

    for (float x : result_data)
    {
        ASSERT_EQ(x, 2);
    }
}

TEST(async, tensor_read)
{
}

TEST(async, tensor_write)
{
}
