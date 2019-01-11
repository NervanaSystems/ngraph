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

#pragma once

#include <functional>
#include <memory>
#include <set>
#include <tuple>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
        class RecurrentGraphRewrite;
    }

    using graph_rewrite_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
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

class ngraph::pass::GraphRewrite : public FunctionPass
{
public:
    GraphRewrite()
        : FunctionPass()
    {
    }

    bool is_enabled(std::shared_ptr<pattern::Matcher> m);
    void add_matcher(std::shared_ptr<pattern::Matcher> m,
                     const ngraph::graph_rewrite_callback& callback);
    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);

private:
    std::vector<std::tuple<std::shared_ptr<pattern::Matcher>, ngraph::graph_rewrite_callback>>
        m_matchers;
};

class ngraph::pass::RecurrentGraphRewrite : public FunctionPass
{
public:
    RecurrentGraphRewrite(size_t num_iters = 10)
        : FunctionPass()
        , m_num_iters(num_iters)
    {
    }

    void add_matcher(std::shared_ptr<pattern::RecurrentMatcher> m) { m_matchers.push_back(m); }
    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);

private:
    size_t m_num_iters;
    std::vector<std::shared_ptr<pattern::RecurrentMatcher>> m_matchers;
};
