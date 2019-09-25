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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
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

#ifdef _WIN32
#include "windows.h"
#define usleep(a) Sleep(a / 1000)
#endif
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

#if defined(NGRAPH_INTERPRETER_ENABLE)
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
#endif

TEST(util, traverse_functions)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C}, "f");

    vector<Function*> functions;
    traverse_functions(f, [&](shared_ptr<Function> fp) { functions.push_back(fp.get()); });
    ASSERT_EQ(1, functions.size());
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
        make_shared<Function>(AplusBtimesC, ParameterVector{A, B, C}, "f");

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
            if (*cloneit != nm.at((*origit).get()))
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

    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(A.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(B.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Parameter>(node_map.at(C.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Add>(node_map.at(AplusB.get())));
    ASSERT_NE(nullptr, as_type_ptr<op::Multiply>(node_map.at(AplusBtimesC.get())));

    auto sorted_nodes = topological_sort(nodes);
    auto sorted_cloned_nodes = topological_sort(cloned_nodes);
    ASSERT_TRUE(CompareNodeVector(sorted_nodes, sorted_cloned_nodes, node_map));
}

TEST_F(CloneTest, clone_nodes_partial)
{
    // map A -> A' prior to clone
    auto Aprime = make_shared<op::Parameter>(element::f32, shape);
    node_map[A.get()] = Aprime;

    auto cloned_nodes = clone_nodes(nodes, node_map);
    ASSERT_TRUE(CompareNodeVector(nodes, cloned_nodes, node_map));

    // ensure A -> A' after clone
    ASSERT_EQ(Aprime, node_map.at(A.get()));
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

    auto f = make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, ParameterVector{A, B, C});

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

TEST(graph_util, get_subgraph_outputs_trivial_tests)
{
    auto outputs = ngraph::get_subgraph_outputs(NodeVector{}, NodeVector{});
    ASSERT_EQ(outputs.size(), 0);

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto absn = make_shared<op::Abs>(A);
    auto neg_absn = make_shared<op::Negative>(absn);
    outputs = ngraph::get_subgraph_outputs(NodeVector{A}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{A}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{A}, NodeVector{A});
    ASSERT_EQ(outputs, (NodeVector{}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{A, absn}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{absn}));

    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto abs_b = make_shared<op::Abs>(B);
    auto neg_b = make_shared<op::Negative>(B);
    auto abs_b_neg = make_shared<op::Negative>(abs_b);
    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B, abs_b}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b}, NodeVector{B});
    ASSERT_EQ(outputs, (NodeVector{abs_b}));

    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b, abs_b_neg}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B}));

    auto add_b = make_shared<op::Add>(neg_b, abs_b_neg);
    outputs =
        ngraph::get_subgraph_outputs(NodeVector{B, abs_b, neg_b, abs_b_neg, add_b}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{}));

    // now add_b uses abs_b_neg
    outputs = ngraph::get_subgraph_outputs(NodeVector{B, abs_b, abs_b_neg}, NodeVector{});
    ASSERT_EQ(outputs, (NodeVector{B, abs_b_neg}));
}

TEST(util, test_fprop_cache)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto output = (A + B) * C + A;

    auto f = make_shared<Function>(NodeVector{output}, ParameterVector{A, B, C});

    auto bf = autodiff::backprop_function(f);

    auto fprop_cache = cache_fprop(f, bf);

    EXPECT_EQ(fprop_cache.fprop->get_results().size(), 2);
    EXPECT_EQ(fprop_cache.bprop->get_parameters().size(), 5);
}

TEST(graph_util, test_subgraph_topological_sort)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto add = A + B;
    auto mul = C * add;
    auto result = make_shared<op::Result>(mul);
    auto sorted = ngraph::subgraph_topological_sort(NodeVector{mul, add, A});
    std::list<std::shared_ptr<Node>> expected{A, add, mul};
    ASSERT_EQ(expected, sorted);
}

TEST(graph_util, test_subgraph_topological_sort_control_dependencies)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Abs>(A);
    auto E = make_shared<op::Abs>(B);
    auto add = A + B;
    add->add_control_dependency(D);
    add->add_control_dependency(E);
    auto mul = C * add;
    auto result = make_shared<op::Result>(mul);
    auto sorted = ngraph::subgraph_topological_sort(NodeVector{mul, add, A, D}, true);
    std::list<std::shared_ptr<Node>> expected{A, D, add, mul};
    ASSERT_EQ(expected, sorted);
}

TEST(util, enum_mask_construction)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    {
        EnumMask<Type> m;
        EXPECT_EQ(0, m.value());
    }
    {
        EnumMask<Type> m(Type::c);
        EXPECT_EQ(static_cast<uint32_t>(Type::c), m.value());
    }
    {
        EnumMask<Type> a(Type::c);
        EnumMask<Type> b{a};
        EXPECT_EQ(a.value(), b.value());
    }
    {
        EnumMask<Type> a{Type::a, Type::c, Type::d};
        EXPECT_EQ((static_cast<uint32_t>(Type::a) | static_cast<uint32_t>(Type::c) |
                   static_cast<uint32_t>(Type::d)),
                  a.value());
    }
}

TEST(util, enum_mask_set_clear)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    EnumMask<Type> m;
    m.set(Type::b);
    EXPECT_EQ(static_cast<uint32_t>(Type::b), m.value());
    m.set(Type::c);
    EXPECT_EQ(static_cast<uint32_t>(Type::b) | static_cast<uint32_t>(Type::c), m.value());
    m.clear(Type::b);
    EXPECT_EQ(static_cast<uint32_t>(Type::c), m.value());
    m.clear_all();
    EXPECT_EQ(0, m.value());
    m.set(Type::d);
    m.set(Type::b);
    EXPECT_EQ(true, m.is_set(Type::d));
    EXPECT_EQ(false, m.is_set(Type::a));
    EXPECT_EQ(true, m.is_set(Type::b));
    EXPECT_EQ(false, m.is_set(Type::c));
    EXPECT_EQ(false, m.is_set({Type::a, Type::b}));
    EXPECT_EQ(false, m.is_set({Type::c, Type::d}));
    EXPECT_EQ(false, m.is_set({Type::a, Type::c}));
    EXPECT_EQ(true, m.is_set({Type::b, Type::d}));
    EXPECT_EQ(false, m.is_clear(Type::d));
    EXPECT_EQ(true, m.is_clear(Type::a));
    EXPECT_EQ(false, m.is_clear(Type::b));
    EXPECT_EQ(true, m.is_clear(Type::c));
    EXPECT_EQ(false, m.is_clear({Type::c, Type::d}));
    EXPECT_EQ(false, m.is_clear({Type::a, Type::b}));
    EXPECT_EQ(true, m.is_clear({Type::a, Type::c}));
    EXPECT_EQ(false, m.is_clear({Type::b, Type::d}));

    EXPECT_EQ(true, m.is_any_set({Type::a, Type::b}));
    EXPECT_EQ(true, m.is_any_set({Type::a, Type::d}));
    EXPECT_EQ(true, m.is_any_set({Type::b, Type::c}));
    EXPECT_EQ(true, m.is_any_set({Type::c, Type::d}));
    EXPECT_EQ(false, m.is_any_set({Type::a, Type::c}));
    EXPECT_EQ(true, m.is_any_clear({Type::c, Type::d}));
    EXPECT_EQ(true, m.is_any_clear({Type::a, Type::b}));
    EXPECT_EQ(true, m.is_any_clear({Type::a, Type::c}));
    EXPECT_EQ(true, m.is_any_clear({Type::b, Type::c}));
    EXPECT_EQ(false, m.is_any_clear({Type::b, Type::d}));

    m.set(Type::a);
    EXPECT_EQ(false, m.is_clear(Type::a));
    EXPECT_EQ(false, m.is_clear(Type::b));
    EXPECT_EQ(true, m.is_clear(Type::c));
    EXPECT_EQ(false, m.is_clear(Type::d));
}

TEST(util, enum_mask_operators)
{
    enum class Type : uint32_t
    {
        a = 0x1,
        b = 1 << 1,
        c = 1 << 2,
        d = 1 << 3
    };
    EnumMask<Type> m;
    m = Type::b;
    EXPECT_EQ(static_cast<uint32_t>(Type::b), m.value());
    EXPECT_EQ(true, m[Type::b]);
    EXPECT_EQ(false, m[Type::a]);
    EXPECT_EQ(false, m[Type::c]);
    m |= Type::c;
    EXPECT_EQ(static_cast<uint32_t>(Type::b) | static_cast<uint32_t>(Type::c), m.value());
    m &= Type::d;
    EXPECT_EQ(0, m.value());

    m |= Type::a;
    m |= Type::c;
    EXPECT_EQ(true, m.is_set(Type::a));
    EXPECT_EQ(false, m.is_set(Type::b));
    EXPECT_EQ(true, m.is_set(Type::c));
    EXPECT_EQ(false, m.is_set(Type::d));
    EXPECT_EQ(true, m.is_any_set(Type::a));
    EXPECT_EQ(false, m.is_any_set(Type::b));
    EXPECT_EQ(true, m.is_any_set(Type::c));
    EXPECT_EQ(false, m.is_any_set(Type::d));
    EXPECT_EQ(true, m.is_any_set({Type::a, Type::c}));
    EXPECT_EQ(false, m.is_any_set({Type::b, Type::d}));

    EnumMask<Type> n;
    n = m | n;
    EXPECT_EQ(m, n);
    n = m & n;
    EXPECT_EQ(m, n);
    bool r = (n == m);
    EXPECT_EQ(true, r);
    r = (n != m);
    EXPECT_EQ(false, r);
    n.clear_all();
    n = {Type::a, Type::b};
    r = (n == m);
    EXPECT_EQ(false, r);
    r = (n != m);
    EXPECT_EQ(true, r);
    n = m & n;
    EXPECT_EQ(static_cast<uint32_t>(Type::a), n.value());
    n = m | Type::b;
    EXPECT_EQ(true, n.is_set(Type::a));
    EXPECT_EQ(true, n.is_set(Type::b));
    EXPECT_EQ(true, n.is_set(Type::c));
    EXPECT_EQ(false, n.is_set(Type::d));
    EXPECT_EQ(false, n[Type::d]);
    EXPECT_EQ(true, n[Type::b]);
}

TEST(graph, huge)
{
    std::vector<std::weak_ptr<Node>> weak_nodes;
    {
        auto param = make_shared<op::Parameter>(element::f32, Shape{3, 3});
        std::shared_ptr<Node> n = param;
        for (size_t i = 0; i < 1000000; i++)
        {
            n = make_shared<op::Negative>(n);
        }
        auto f = make_shared<Function>(NodeVector{n}, ParameterVector{param});
        f->map_unordered_ops(
            [&weak_nodes](Node* node) { weak_nodes.push_back(node->shared_from_this()); });
    }

    for (auto& weak_node : weak_nodes)
    {
        EXPECT_TRUE(weak_node.expired());
    }
}

TEST(util, apply_permutation)
{
    ASSERT_EQ(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{2, 1, 0, 3}), (Shape{2, 1, 0, 3}));
}

TEST(util, apply_permutation_too_short_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2}), CheckFailure);
}

TEST(util, apply_permutation_too_long_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 3, 3}), CheckFailure);
}

TEST(util, apply_permutation_oob_axis_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 4}), CheckFailure);
}

TEST(util, apply_permutation_repeated_axis_fails)
{
    ASSERT_THROW(apply_permutation(Shape{0, 1, 2, 3}, AxisVector{0, 1, 2, 2}), CheckFailure);
}

TEST(util, apply_permutation_pshape)
{
    ASSERT_TRUE(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{2, 1, 0, 3})
            .same_scheme(PartialShape{2, Dimension::dynamic(), 0, 3}));
}

TEST(util, apply_permutation_pshape_rank_dynamic)
{
    ASSERT_TRUE(apply_permutation(PartialShape::dynamic(), AxisVector{2, 1, 0, 3})
                    .same_scheme(PartialShape::dynamic()));
}

TEST(util, apply_permutation_pshape_too_short_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_too_long_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 3, 3}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_oob_axis_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 4}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_repeated_axis_fails)
{
    ASSERT_THROW(
        apply_permutation(PartialShape{0, Dimension::dynamic(), 2, 3}, AxisVector{0, 1, 2, 2}),
        CheckFailure);
}

TEST(util, apply_permutation_pshape_rank_dynamic_inviable_permutation_fails)
{
    ASSERT_THROW(apply_permutation(PartialShape::dynamic(), AxisVector{0, 1, 2, 2}), CheckFailure);
}

TEST(util, clone_function_friendly_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    A->set_friendly_name("A");
    B->set_friendly_name("B");

    auto g = clone_function(*f);

    bool found_A = false;
    bool found_B = false;
    for (auto parameter : g->get_parameters())
    {
        found_A |= parameter->get_friendly_name() == "A";
        found_B |= parameter->get_friendly_name() == "B";
    }
    EXPECT_TRUE(found_A);
    EXPECT_TRUE(found_B);
}
