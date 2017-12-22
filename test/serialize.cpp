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

#include "ngraph/file_util.hpp"
#include "ngraph/json.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(serialize, tuple)
{
    auto shape = Shape{2, 2};
    auto tensor_view_type = make_shared<TensorViewType>(element::i64, shape);

    auto A = make_shared<op::Parameter>(tensor_view_type);
    auto B = make_shared<op::Parameter>(tensor_view_type);
    auto C = make_shared<op::Parameter>(tensor_view_type);

    auto ttt =
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type});

    auto f = make_shared<XLAFunction>(
        make_shared<op::XLATuple>(Nodes{(A + B), (A - B), (C * A)}), ttt, op::Parameters{A, B, C});

    string js = serialize(f, 4);
    {
        ofstream f("serialize_function_tuple.js");
        f << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);
}

TEST(serialize, main)
{
    // First create "f(A,B,C) = (A+B)*C".
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto rt_f = make_shared<TensorViewType>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, rt_f, op::Parameters{A, B, C}, "f");

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto rt_g = make_shared<TensorViewType>(element::f32, shape);
    auto g = make_shared<Function>(make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}) +
                                       make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}),
                                   rt_g,
                                   op::Parameters{X, Y, Z},
                                   "g");

    // Now make "h(X,Y,Z) = g(X,Y,Z) + g(X,Y,Z)"
    auto X1 = make_shared<op::Parameter>(element::f32, shape);
    auto Y1 = make_shared<op::Parameter>(element::f32, shape);
    auto Z1 = make_shared<op::Parameter>(element::f32, shape);
    auto rt_h = make_shared<TensorViewType>(element::f32, shape);
    auto h = make_shared<Function>(make_shared<op::FunctionCall>(g, Nodes{X1, Y1, Z1}) +
                                       make_shared<op::FunctionCall>(g, Nodes{X1, Y1, Z1}),
                                   rt_h,
                                   op::Parameters{X1, Y1, Z1},
                                   "h");

    string js = serialize(h, 4);

    {
        ofstream f("serialize_function.js");
        f << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);

    // Now call g on some test vectors.
    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(sfunc);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto x = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({x, y, z}, {result});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({y, x, z}, {result});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({x, z, y}, {result});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector<float>());
}

TEST(serialize, existing_models)
{
    vector<string> models = {"mxnet/mnist_mlp_forward.json", "mxnet/10_bucket_LSTM.json"};

    for (const string& model : models)
    {
        const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
        const string json_string = file_util::read_file_to_string(json_path);
        stringstream ss(json_string);
        shared_ptr<Function> f = ngraph::deserialize(ss);
    }
}
