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

#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(util, split)
{
    {
        string s1 = "this,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(4, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("is", r1[1].c_str());
        EXPECT_STRCASEEQ("a", r1[2].c_str());
        EXPECT_STRCASEEQ("test", r1[3].c_str());
    }

    {
        string s1 = "this,is,a,test,";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("is", r1[1].c_str());
        EXPECT_STRCASEEQ("a", r1[2].c_str());
        EXPECT_STRCASEEQ("test", r1[3].c_str());
        EXPECT_STRCASEEQ("", r1[4].c_str());
    }

    {
        string s1 = ",this,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("", r1[0].c_str());
        EXPECT_STRCASEEQ("this", r1[1].c_str());
        EXPECT_STRCASEEQ("is", r1[2].c_str());
        EXPECT_STRCASEEQ("a", r1[3].c_str());
        EXPECT_STRCASEEQ("test", r1[4].c_str());
    }

    {
        string s1 = "this,,is,a,test";
        auto r1 = split(s1, ',');
        ASSERT_EQ(5, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
        EXPECT_STRCASEEQ("", r1[1].c_str());
        EXPECT_STRCASEEQ("is", r1[2].c_str());
        EXPECT_STRCASEEQ("a", r1[3].c_str());
        EXPECT_STRCASEEQ("test", r1[4].c_str());
    }

    {
        string s1 = "this";
        auto r1 = split(s1, ',');
        ASSERT_EQ(1, r1.size());
        EXPECT_STRCASEEQ("this", r1[0].c_str());
    }

    {
        string s1 = "";
        auto r1 = split(s1, ',');
        ASSERT_EQ(1, r1.size());
        EXPECT_STRCASEEQ("", r1[0].c_str());
    }
}

TEST(DISABLED_util, dump)
{
    string text = "this is a text string used to test the dump function.";

    dump(cout, text.data(), text.size());
}

TEST(util, stopwatch)
{
    stopwatch t1;

    t1.start();
    usleep(1000);
    t1.stop();

    t1.start();
    usleep(1000);
    t1.stop();

    t1.start();
    usleep(1000);
    t1.stop();

    EXPECT_EQ(3, t1.get_call_count());

    EXPECT_GT(t1.get_total_microseconds(), t1.get_microseconds());
}

TEST(util, trim)
{
    EXPECT_STREQ("test", trim("test").c_str());
    EXPECT_STREQ("test", trim(" test").c_str());
    EXPECT_STREQ("test", trim("test ").c_str());
    EXPECT_STREQ("test", trim(" test ").c_str());
    EXPECT_STREQ("test", trim("           test            ").c_str());
    EXPECT_STREQ("test", trim("\ttest").c_str());
    EXPECT_STREQ("test", trim("test\t").c_str());
    EXPECT_STREQ("test", trim("\ttest\t").c_str());
    EXPECT_STREQ("test", trim(" \t test \t ").c_str());
}

TEST(util, contains)
{
    vector<int> v1 = {1, 2, 3, 4, 5, 6};

    EXPECT_TRUE(contains(v1, 1));
    EXPECT_TRUE(contains(v1, 4));
    EXPECT_TRUE(contains(v1, 6));
    EXPECT_FALSE(contains(v1, 8));
}

TEST(util, remove_from)
{
}

TEST(util, reduce)
{
    {
        std::vector<size_t> x = {};
        size_t actual =
            ngraph::reduce(x.begin(), x.end(), [](size_t a, size_t b) { return a + b; });
        EXPECT_EQ(actual, 0);
    }
    {
        std::vector<size_t> x = {10};
        size_t actual =
            ngraph::reduce(x.begin(), x.end(), [](size_t a, size_t b) { return a + b; });
        EXPECT_EQ(actual, 10);
    }
    {
        std::vector<size_t> x = {1, 2, 3, 4, 5, 6};
        size_t actual =
            ngraph::reduce(x.begin(), x.end(), [](size_t a, size_t b) { return a + b; });
        EXPECT_EQ(actual, 21);
    }
    {
        std::vector<size_t> x = {1, 2, 3, 4, 5, 6};
        size_t actual = ngraph::reduce(x.begin(), x.end(), ngraph::plus<size_t>);
        EXPECT_EQ(actual, 21);
    }
    {
        std::vector<size_t> x = {1, 2, 3, 4, 5, 6};
        size_t actual = ngraph::reduce(x.begin(), x.end(), ngraph::mul<size_t>);
        EXPECT_EQ(actual, 720);
    }
}

TEST(util, all_close)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{2, 3});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{2, 3});

    copy_data(a, runtime::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());
    copy_data(b, runtime::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_TRUE(ngraph::test::all_close<float>(a, b));

    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{2, 3});
    copy_data(c, runtime::NDArray<float, 2>({{1.1f, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, 0, .05f));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, 0, .11f));

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, .05f, 0));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, .11f, 0));
}

TEST(util, traverse_functions)
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

    vector<Function*> functions;
    traverse_functions(h, [&](shared_ptr<Function> fp) { functions.push_back(fp.get()); });
    ASSERT_EQ(3, functions.size());
}
