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

#include <fstream>
#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using json = nlohmann::json;

using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::StrEq;

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
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C}, "f");

    string js = serialize(f, 4);

    {
        ofstream out("serialize_function.js");
        out << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);
    auto backend = runtime::Backend::create("INTERPRETER");
    auto handle = backend->compile(sfunc);

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    handle->call_with_validate({result}, {x, y, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {y, x, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {x, z, y});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), read_vector<float>(result));
}

TEST(serialize, friendly_name)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto sum = A + B;
    auto product = sum * C;
    auto f = make_shared<Function>(product, ParameterVector{A, B, C}, "f");

    A->set_friendly_name("A");
    B->set_friendly_name("B");
    C->set_friendly_name("C");
    sum->set_friendly_name("Sum");
    product->set_friendly_name("Product");

    string js = serialize(f, 4);
    ofstream out("serialize_function.js");
    out << js;

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);
    auto backend = runtime::Backend::create("INTERPRETER");
    auto handle = backend->compile(sfunc);

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    handle->call_with_validate({result}, {x, y, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {y, x, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {x, z, y});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), read_vector<float>(result));
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
    auto f = make_shared<Function>(A, ParameterVector{});

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

    ngraph::set_serialize_output_shapes(true);
    ofstream out("test.json");
    out << serialize(f, 4);
}

MATCHER_P2(IsOutputShape, type, shape, "")
{
    return std::get<0>(arg) == type && std::get<1>(arg).to_shape() == shape;
}

TEST(serialize, passthrough)
{
    const string tmp_file = "serialize_passthrough.json";

    using estuple = std::tuple<element::Type, PartialShape>;

    Shape shape{2, 2, 2};
    auto p = make_shared<op::Passthrough>(
        "SerializationTest",
        "Plain",
        "Hello, world!",
        NodeVector{},
        std::vector<estuple>{estuple{element::f32, PartialShape{2, 3}},
                             estuple{element::i8, PartialShape{4, 5}}});
    auto f = make_shared<Function>(NodeVector{std::make_shared<op::GetOutputElement>(p, 0),
                                              std::make_shared<op::GetOutputElement>(p, 1)},
                                   ParameterVector{});
    serialize(tmp_file, f);

    auto g = deserialize(tmp_file);
    file_util::remove_file(tmp_file);
    ASSERT_THAT(g, NotNull());

    std::shared_ptr<op::Passthrough> pt;
    for (const auto& op : g->get_ops())
    {
        pt = dynamic_pointer_cast<op::Passthrough>(op);
        if (pt)
        {
            break;
        }
    }
    ASSERT_THAT(pt.get(), NotNull());

    EXPECT_THAT(pt->logical_type(), StrEq("SerializationTest"));
    EXPECT_THAT(pt->language(), StrEq("Plain"));
    EXPECT_THAT(pt->function(), StrEq("Hello, world!"));
    EXPECT_THAT(pt->output_shapes(),
                ElementsAre(IsOutputShape(element::f32, Shape{2, 3}),
                            IsOutputShape(element::i8, Shape{4, 5})));
}

TEST(serialize, constant_infinity_nan)
{
    vector<float> a_data{123.f, 456.f, INFINITY, -INFINITY, NAN};
    vector<float> b_data{5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
    vector<float> c_data{0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05001f, 0.05f};
    vector<int64_t> d_data{-100, -10, -1, 0, 50, 5000000000001};
    auto A = make_shared<op::Constant>(element::f32, Shape{5}, a_data);
    auto B = make_shared<op::Constant>(element::f32, Shape{6}, b_data);
    auto C = make_shared<op::Constant>(element::f32, Shape{7}, c_data);
    auto D = make_shared<op::Constant>(element::i64, Shape{d_data.size()}, d_data);
    A->set_friendly_name("A");
    B->set_friendly_name("B");
    C->set_friendly_name("C");
    D->set_friendly_name("D");
    auto f = make_shared<Function>(NodeVector{A, B, C, D}, ParameterVector{});

    string s = serialize(f, 4);
    shared_ptr<Function> g = deserialize(s);

    shared_ptr<op::Constant> a;
    shared_ptr<op::Constant> b;
    shared_ptr<op::Constant> c;
    shared_ptr<op::Constant> d;
    for (auto node : g->get_ops())
    {
        if (node->get_friendly_name() == "A")
        {
            a = static_pointer_cast<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "B")
        {
            b = static_pointer_cast<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "C")
        {
            c = static_pointer_cast<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "D")
        {
            d = static_pointer_cast<op::Constant>(node);
        }
    }
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(d, nullptr);
    EXPECT_TRUE(test::all_close_f(a->get_vector<float>(), a_data));
    EXPECT_TRUE(test::all_close_f(b->get_vector<float>(), b_data));
    EXPECT_TRUE(test::all_close_f(c->get_vector<float>(), c_data));
    EXPECT_EQ(d->get_vector<int64_t>(), d_data);

    string filename = "constant_infinity_nan_test.dot";
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>(filename);
    pass_manager.run_passes(g);
    ifstream file(filename);
    ASSERT_TRUE(file);
    string str((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    EXPECT_NE(str.find(R"(label="A)"), string::npos);
    EXPECT_NE(str.find(R"(label="B)"), string::npos);
    EXPECT_NE(str.find(R"(label="C)"), string::npos);
    EXPECT_NE(str.find(R"(label="D)"), string::npos);
}

TEST(serialize, non_zero_node_output)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{10});
    auto topk = make_shared<op::TopK>(arg, 0, element::i32, 5, true);
    auto abs = make_shared<op::Abs>(Output<Node>(topk, 1));
    auto result = make_shared<op::Result>(abs);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_abs = g_result->input(0).get_source_output().get_node_shared_ptr();
    auto topk_out = g_abs->input(0).get_source_output();
    EXPECT_EQ(topk_out.get_index(), 1);
    EXPECT_EQ(topk_out.get_node()->description(), "TopK");
}
