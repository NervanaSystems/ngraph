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

#include <memory>
#include <algorithm>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"

class TestMatcher : public ngraph::pattern::Matcher
{
public:
    bool compare_nodes(std::shared_ptr<ngraph::Node>& pattern_node,
                       const std::shared_ptr<ngraph::Node>& graph_node);

    virtual void on_match_class(std::shared_ptr<ngraph::Node>& pattern_node,
                                const std::shared_ptr<ngraph::Node>& graph_node,
                                bool is_match);

    std::shared_ptr<ngraph::Node> get_pattern_node() const { return m_pattern_node; }
    std::shared_ptr<ngraph::Node> get_graph_node() const { return m_graph_node; }
    bool get_is_valid() const { return m_is_valid; }
protected:
    bool m_is_valid;
    bool m_is_match;
    std::shared_ptr<ngraph::Node> m_pattern_node;
    std::shared_ptr<ngraph::Node> m_graph_node;
};

void TestMatcher::on_match_class(std::shared_ptr<ngraph::Node>& pattern_node,
                                 const std::shared_ptr<ngraph::Node>& graph_node,
                                 bool is_match)
{
    m_pattern_node = pattern_node;
    m_graph_node = graph_node;
    m_is_match = is_match;
    m_is_valid = true;
}


void ngraph::pattern::Matcher::match_arguments(const Nodes& pattern_args, const Nodes& args)
{
    for (size_t i = 0; i < args.size(); i++)
    {
        pattern_args.at(i)->match_class(*this, args.at(i));
        if (!m_is_match)
        {
            return;
        }
    }
}

void ngraph::pattern::Matcher::on_match_class(std::shared_ptr<ngraph::Node>& pattern_node,
    const std::shared_ptr<ngraph::Node>& graph_node,
    bool is_match)
{
    if (!is_match)
    {
        m_is_match = false;
        return;
    }

    auto args = graph_node->get_arguments();
    auto pattern_args = pattern_node->get_arguments();

    if (args.size() != pattern_args.size())
    {
        m_is_match = false;
        return;
    }

    if (graph_node->is_commutative())
    {

        auto args_copy = Nodes(args); //@TODO [nikolayk] remove if there are no implicit dependencies
        do                            //on the order of arguments in the rest of the compiler
        {
            m_is_match = true; //in case if set to false by the previous permutation
            match_arguments(pattern_args, args_copy);
            if (m_is_match)
            {
                return;
            }
        } while (std::next_permutation(begin(args_copy), end(args_copy)));
    }
    else
    {
        match_arguments(pattern_args, args);
    }
}

bool ngraph::pattern::Matcher::match(std::shared_ptr<Node>& pattern_node, const std::shared_ptr<Node>& graph_node)
{
    m_is_valid = true;
    pattern_node->match_class(*this, graph_node);
    return m_is_match;
}

bool TestMatcher::compare_nodes(std::shared_ptr<ngraph::Node>& pattern_node,
                                const std::shared_ptr<ngraph::Node>& graph_node)
{
    m_pattern_node = nullptr;
    m_graph_node = nullptr;
    m_is_match = false;
    m_is_valid = false;
    pattern_node->match_class(*this, graph_node);
    return m_is_match;
}

using namespace ngraph;
using namespace std;

TEST(pattern, op_op)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto SUM = A + B;
    auto ANY = std::make_shared<pattern::op::Any>();

    TestMatcher test_matcher;
    ASSERT_TRUE(test_matcher.compare_nodes(A, B));
    ASSERT_FALSE(test_matcher.compare_nodes(A, SUM));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, A));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, B));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, SUM));
}
