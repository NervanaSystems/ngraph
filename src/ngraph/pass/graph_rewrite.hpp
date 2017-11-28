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

#pragma once

#include <functional>
#include <set>
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
    }
    namespace pattern
    {
        class Matcher;
    }
}

/// \brief GraphRewrite (in tandem with \sa Matcher) performs transformations on specified patterns
///
/// Graph rewrite pass essentially allows pass users to rewrite parts of the
/// input graph in any way they want. Fusion is one example of graph rewrite that
/// fuses multiple ops together. At a high-level users of the pass need to
/// specify 2 things: 1) which ops to fuse (via \sa Matcher, and 2) how to create new op(s) from
/// the existing ops by providing a callback to \p Matcher object
/// Patterns can be added by using \sa add_matcher
/// Callbacks should use \sa replace_node to transform matched sub graphs

class ngraph::pass::GraphRewrite : public CallGraphPass
{
public:
    GraphRewrite()
        : CallGraphPass()
    {
    }

    void add_matcher(std::shared_ptr<pattern::Matcher> m) { m_matchers.push_back(m); }
    static void replace_node_users_arguments(std::shared_ptr<Node> target,
                                             std::shared_ptr<Node> replacement);
    static void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);
    virtual bool run_on_call_graph(std::list<std::shared_ptr<ngraph::Node>>&) override;

private:
    //enable cascading rewrites
    std::vector<std::shared_ptr<pattern::Matcher>> m_matchers;
};
