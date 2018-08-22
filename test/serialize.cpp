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

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using json = nlohmann::json;

template <typename T>
T get_or_default(nlohmann::json& j, const std::string& key, const T& default_value)
{
    T rc;
    try
    {
        rc = j.at(key).get<T>();
    }
    catch (...)
    {
        rc = default_value;
    }
    return rc;
}

#if defined(NGRAPH_INTERPRETER_ENABLE)
TEST(serialize, main)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C}, "f");

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto g = make_shared<Function>(make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}) +
                                       make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}),
                                   op::ParameterVector{X, Y, Z},
                                   "g");

    // Now make "h(X,Y,Z) = g(X,Y,Z) + g(X,Y,Z)"
    auto X1 = make_shared<op::Parameter>(element::f32, shape);
    auto Y1 = make_shared<op::Parameter>(element::f32, shape);
    auto Z1 = make_shared<op::Parameter>(element::f32, shape);
    auto h = make_shared<Function>(make_shared<op::FunctionCall>(g, NodeVector{X1, Y1, Z1}) +
                                       make_shared<op::FunctionCall>(g, NodeVector{X1, Y1, Z1}),
                                   op::ParameterVector{X1, Y1, Z1},
                                   "h");

    string js = serialize(h, 4);

    {
        ofstream out("serialize_function.js");
        out << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);

    // Now call h on some test vectors.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(sfunc, {result}, {x, y, z});
    EXPECT_EQ((vector<float>{216, 320, 440, 576}), read_vector<float>(result));

    backend->call_with_validate(sfunc, {result}, {y, x, z});
    EXPECT_EQ((vector<float>{216, 320, 440, 576}), read_vector<float>(result));

    backend->call_with_validate(sfunc, {result}, {x, z, y});
    EXPECT_EQ((vector<float>{200, 288, 392, 512}), read_vector<float>(result));
}
#endif

TEST(serialize, existing_models)
{
    vector<string> models = {"mxnet/mnist_mlp_forward.json",
                             "mxnet/10_bucket_LSTM.json",
                             "mxnet/LSTM_backward.json",
                             "mxnet/LSTM_forward.json"};

    for (const string& model : models)
    {
        const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
        const string json_string = file_util::read_file_to_string(json_path);
        shared_ptr<Function> f = ngraph::deserialize(json_string);
    }
}

TEST(serialize, default_value)
{
    json j = {{"test1", 1}, {"test2", 2}};

    int x1 = j.at("test1").get<int>();
    EXPECT_EQ(x1, 1);
    int x2 = get_or_default<int>(j, "test2", 0);
    EXPECT_EQ(x2, 2);
    int x3 = get_or_default<int>(j, "test3", 3);
    EXPECT_EQ(x3, 3);
}

TEST(serialize, constant)
{
    const string tmp_file = "serialize_constant.cpio";
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, op::ParameterVector{});

    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), A->get_vector<float>());
    serialize(tmp_file, f);
    auto g = deserialize(tmp_file);
    ASSERT_NE(g, nullptr);
    file_util::remove_file(tmp_file);
    bool found = false;
    for (shared_ptr<Node> node : g->get_ops())
    {
        shared_ptr<op::Constant> c = dynamic_pointer_cast<op::Constant>(node);
        if (c)
        {
            found = true;
            EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), c->get_vector<float>());
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(benchmark, serialize)
{
    stopwatch timer;
    string model = "mxnet/LSTM_backward.json";

    const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
    timer.start();
    const string json_string = file_util::read_file_to_string(json_path);
    timer.stop();
    cout << "file read took " << timer.get_milliseconds() << "ms\n";
    timer.start();
    shared_ptr<Function> f = ngraph::deserialize(json_string);
    timer.stop();
    cout << "deserialize took " << timer.get_milliseconds() << "ms\n";
}
