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

#include <sstream>
#include <string>
#include <set>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class VisualizeTree;
    }
    class Node;
}

class ngraph::pass::VisualizeTree : public FunctionPass
{
public:
    VisualizeTree(const std::string& file_name);
    bool run_on_function(ngraph::Function*) override;

private:
    std::string add_attributes(const Node* node);
    std::string get_attributes(const Node* node);
    void render() const;

    std::stringstream     m_ss;
    std::string           m_name;
    std::set<const Node*> m_nodes_with_attributes;
};
