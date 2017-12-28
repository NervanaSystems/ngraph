// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <vector>

#include "gtest/gtest.h"

#include "ngraph/builder/xla_tuple.hpp"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(builder_xla, simple)
{
    auto shape = Shape{2, 2};

    auto pA = make_shared<op::Parameter>(element::f32, shape);
    auto pB = make_shared<op::Parameter>(element::f32, shape);
    auto pC = make_shared<op::Parameter>(element::f32, shape);

    auto ABC = make_shared<builder::Tuple>(Nodes{pA, pB, pC});

    auto A = builder::get_tuple_element(ABC, 0);
    auto B = builder::get_tuple_element(ABC, 1);
    auto C = builder::get_tuple_element(ABC, 2);
    auto f = make_shared<Function>(Nodes{(A + B) * C}, op::Parameters{pA, pB, pC});

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
    //auto abc = runtime::make_tuple({a, b, c});
    //auto bac = runtime::make_tuple({b, a, c});
    //auto acb = runtime::make_tuple({a, c, b});
    auto result = backend->make_primary_tensor_view(element::f32, shape);
    //auto result_tuple = runtime::make_tuple({result});

    //cf->call({abc}, {result_tuple});
    cf->call({a, b, c}, {result});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    //cf->call({bac}, {result_tuple});
    cf->call({b, a, c}, {result});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    //cf->call({acb}, {result_tuple});
    cf->call({a, c, b}, {result});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector<float>());
}
