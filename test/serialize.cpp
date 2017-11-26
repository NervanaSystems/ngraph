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

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"

#include "ngraph/json.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

TEST(serialize, element_type)
{
    nlohmann::json j;
    element::Type input = element::f32;
    j = input;

    element::Type output = j.get<element::Type>();

    EXPECT_EQ(input, output);
}

TEST(serialize, main)
{
    // First create "f(A,B,C) = (A+B)*C".
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_f = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt_f, op::Parameters{A, B, C}, "f");

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Z = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_g = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto g = make_shared<Function>(make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}) +
                                       make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}),
                                   rt_g,
                                   op::Parameters{X, Y, Z},
                                   "g");

    // Now make "h(X,Y,Z) = g(X,Y,Z) + g(X,Y,Z)"
    auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Z1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_h = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto h = make_shared<Function>(make_shared<op::FunctionCall>(g, Nodes{X1, Y1, Z1}) +
                                       make_shared<op::FunctionCall>(g, Nodes{X1, Y1, Z1}),
                                   rt_h,
                                   op::Parameters{X1, Y1, Z1},
                                   "h");

    string js = serialize::serialize(h);

    {
        ofstream f("serialize_function.js");
        f << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = serialize::deserialize(in);

    // Now call g on some test vectors.
    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(sfunc);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
}
