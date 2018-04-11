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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"

using namespace std;
using namespace ngraph;

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

TEST(util, all_close)
{
    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{2, 3});
    auto b = backend->create_tensor(element::f32, Shape{2, 3});

    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{1, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_TRUE(ngraph::test::all_close<float>(a, b));

    auto c = backend->create_tensor(element::f32, Shape{2, 3});
    copy_data(c, test::NDArray<float, 2>({{1.1f, 2, 3}, {3, 4, 5}}).get_vector());

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, 0, .05f));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, 0, .11f));

    EXPECT_FALSE(ngraph::test::all_close<float>(c, a, .05f, 0));
    EXPECT_TRUE(ngraph::test::all_close<float>(c, a, .11f, 0));
}

TEST(util, traverse_functions)
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

    vector<Function*> functions;
    traverse_functions(h, [&](shared_ptr<Function> fp) { functions.push_back(fp.get()); });
    ASSERT_EQ(3, functions.size());
}

class CloneTest : public ::testing::Test
{
public:
    // (A + B) * C
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = A + B;
    std::shared_ptr<Node> AplusBtimesC = AplusB * C;

    NodeMap node_map;
    std::list<std::shared_ptr<ngraph::Node>> nodes;
    std::shared_ptr<Function> func =
        make_shared<Function>(AplusBtimesC, op::ParameterVector{A, B, C}, "f");

    void SetUp()
    {
        nodes.push_back(AplusBtimesC);
        nodes.push_back(AplusB);
        nodes.push_back(A);
        nodes.push_back(B);
        nodes.push_back(C);
    }

    bool CompareNodeVector(const std::list<std::shared_ptr<ngraph::Node>>& orig,
                           const std::list<std::shared_ptr<ngraph::Node>>& clone,
                           const NodeMap& nm)
    {
        if (orig.size() != clone.size())
        {
            return false;
        }
        auto origit = orig.begin();
        auto cloneit = clone.begin();
        while (origit != orig.end() && cloneit != clone.end())
        {
            if (*cloneit != nm.get_node_map().at(*origit))
            {
                return false;
            }
            ++origit;
            ++cloneit;
        }
        return true;
    }
};

TEST_F(CloneTest, clone_nodes_full)
{
    auto cloned_nodes = clone_nodes(nodes, node_map);
    ASSERT_TRUE(CompareNodeVector(nodes, cloned_nodes, node_map));

    ASSERT_NE(nullptr, std::dynamic_pointer_cast<op::Parameter>(node_map.get(A)));
    ASSERT_NE(nullptr, std::dynamic_pointer_cast<op::Parameter>(node_map.get(B)));
    ASSERT_NE(nullptr, std::dynamic_pointer_cast<op::Parameter>(node_map.get(C)));
    ASSERT_NE(nullptr, std::dynamic_pointer_cast<op::Add>(node_map.get(AplusB)));
    ASSERT_NE(nullptr, std::dynamic_pointer_cast<op::Multiply>(node_map.get(AplusBtimesC)));

    auto sorted_nodes = topological_sort(nodes);
    auto sorted_cloned_nodes = topological_sort(cloned_nodes);
    ASSERT_TRUE(CompareNodeVector(sorted_nodes, sorted_cloned_nodes, node_map));
}

TEST_F(CloneTest, clone_nodes_partial)
{
    // map A -> A' prior to clone
    auto Aprime = make_shared<op::Parameter>(element::f32, shape);
    node_map.add(A, Aprime);

    auto cloned_nodes = clone_nodes(nodes, node_map);
    ASSERT_TRUE(CompareNodeVector(nodes, cloned_nodes, node_map));

    // ensure A -> A' after clone
    ASSERT_EQ(Aprime, node_map.get(A));
}

TEST_F(CloneTest, clone_function_full)
{
    auto cloned_func = clone_function(*func, node_map);
    ASSERT_TRUE(CompareNodeVector(func->get_ops(), cloned_func->get_ops(), node_map));
}

TEST(graph_util, clone_multiple_results)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto f =
        make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, op::ParameterVector{A, B, C});

    auto copy = clone_function(*f);
}

TEST(util, round_up)
{
    EXPECT_EQ(0, round_up(0, 4));
    EXPECT_EQ(4, round_up(1, 4));
    EXPECT_EQ(4, round_up(2, 4));
    EXPECT_EQ(4, round_up(3, 4));
    EXPECT_EQ(4, round_up(4, 4));
    EXPECT_EQ(8, round_up(5, 4));
}

TEST(util, parse_string)
{
    EXPECT_FLOAT_EQ(2, parse_string<float>("2"));
    EXPECT_FLOAT_EQ(2.125, parse_string<float>("2.125"));
    EXPECT_FLOAT_EQ(numeric_limits<float>::infinity(), parse_string<float>("INFINITY"));
    EXPECT_FLOAT_EQ(numeric_limits<float>::infinity(), parse_string<float>("infinity"));
    EXPECT_FLOAT_EQ(-numeric_limits<float>::infinity(), parse_string<float>("-INFINITY"));
    EXPECT_TRUE(isnan(parse_string<float>("NaN")));

    EXPECT_FLOAT_EQ(2, parse_string<double>("2"));
    EXPECT_FLOAT_EQ(2.125, parse_string<double>("2.125"));
    EXPECT_FLOAT_EQ(numeric_limits<double>::infinity(), parse_string<double>("INFINITY"));
    EXPECT_FLOAT_EQ(numeric_limits<double>::infinity(), parse_string<double>("infinity"));
    EXPECT_FLOAT_EQ(-numeric_limits<double>::infinity(), parse_string<double>("-INFINITY"));
    EXPECT_TRUE(std::isnan(parse_string<double>("NaN")));
}
