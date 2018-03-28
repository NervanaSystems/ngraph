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

#include <vector>

#include "gtest/gtest.h"

#include "ngraph/builder/xla_tuple.hpp"
#include "ngraph/ngraph.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(builder_xla, simple)
{
    Shape shape{2, 2};

    auto pA = make_shared<op::Parameter>(element::f32, shape);
    auto pB = make_shared<op::Parameter>(element::f32, shape);
    auto pC = make_shared<op::Parameter>(element::f32, shape);

    auto ABC = make_shared<xla::op::Tuple>(NodeVector{pA, pB, pC});

    auto A = xla::op::get_tuple_element(ABC, 0);
    auto B = xla::op::get_tuple_element(ABC, 1);
    auto C = xla::op::get_tuple_element(ABC, 2);
    auto f = make_shared<xla::XLAFunction>(
        NodeVector{make_shared<xla::op::Tuple>(NodeVector{(A + B) * C})}, NodeVector{ABC});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});
    auto abc = xla::make_tuple({a, b, c});
    auto bac = xla::make_tuple({b, a, c});
    auto acb = xla::make_tuple({a, c, b});
    auto result = backend->make_primary_tensor_view(element::f32, shape);
    auto result_tuple = xla::make_tuple({result});

    xla::call(cf, {result_tuple}, {abc});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    xla::call(cf, {result_tuple}, {bac});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    xla::call(cf, {result_tuple}, {acb});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), read_vector<float>(result));
}

TEST(builder_xla, empty_tuple_interpreter)
{
    auto empty_tuple = make_shared<xla::op::Tuple>(NodeVector{});
    auto f = make_shared<xla::XLAFunction>(NodeVector{empty_tuple}, NodeVector{});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    xla::call(cf, {}, {});
}

TEST(builder_xla, empty_tuple_cpu)
{
    auto empty_tuple = make_shared<xla::op::Tuple>(NodeVector{});
    auto f = make_shared<xla::XLAFunction>(NodeVector{empty_tuple}, NodeVector{});

    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    xla::call(cf, {}, {});
}
