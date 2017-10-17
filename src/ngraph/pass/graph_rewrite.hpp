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

    //using gr_callback_fn =
    //    std::function<void(std::shared_ptr<pattern::Matcher> m, std::shared_ptr<Node> match_root, class pass::GraphRewrite& gr)>;
}

class ngraph::pass::GraphRewrite : public CallGraphPass
{
public:
    GraphRewrite()
        : CallGraphPass(){};

    void add_matcher(std::shared_ptr<pattern::Matcher> m)
    {
        m_matchers.push_back(m);
    };

    static void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);
    //virtual bool run_on_call_graph(std::list<Node*>&) override; //stub until @bob fixes run_on_call_graph

    virtual bool run_on_call_graph(std::list<std::shared_ptr<ngraph::Node>>&) override;
    //bool run_on_call_graph(std::list<std::shared_ptr<Node>>& nodes); //this one is being tested

private:
    //enable cascading rewrites
    std::vector<std::shared_ptr<pattern::Matcher>> m_matchers;
};
