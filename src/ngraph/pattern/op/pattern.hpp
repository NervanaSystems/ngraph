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

#include "ngraph/node.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Pattern : public Node
            {
            public:
                Pattern() : Node(), m_binded(nullptr) {};
                virtual std::string description() const { return "pattern"; } //@TODO [nikolayk] edit description to print out if the pattern is binded and if so the binded node
                virtual void propagate_types() {}

                void virtual match_class(ngraph::pattern::Matcher& matcher, std::shared_ptr<Node> graph_node) override;
                bool is_binded() { return (bool)m_binded;  };
                shared_ptr<Node> get_binded_node() { return m_binded; }
            private:
                shared_ptr<Node> m_binded;
            };
        }
    }
}
