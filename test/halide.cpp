//*****************************************************************************
// Copyright 2018 Intel Corporation
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
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(halide, halide_subgraph)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);

    auto relu = make_shared<op::Relu>((A + B) * C);

    auto f = make_shared<Function>(relu + D, ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("CPU");
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> d = backend->create_tensor(element::f32, shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    vector<float> data{-1, 4, -2, 5, 1, 5, 7, 9};

    copy_data(a, data);
    copy_data(b, data);
    copy_data(c, data);
    copy_data(d, data);

    vector<float> expected{1, 36, 6, 55, 3, 55, 105, 171};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});

    EXPECT_TRUE(test::all_close(read_vector<float>(result), expected, 1.0e-4f, 1.0e-4f));
}
