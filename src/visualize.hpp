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
#include <memory>
#include <set>
#include <sstream>

namespace ngraph
{
    class Visualize;
    class Node;
    using node_ptr = std::shared_ptr<Node>;
}

class ngraph::Visualize
{
public:
    Visualize(const std::string& name = "ngraph");

    void add(node_ptr);

    void save_dot(const std::string& path) const;

private:
    std::string add_attributes(const Node* node);
    std::string get_attributes(const Node* node);

    std::stringstream     m_ss;
    std::string           m_name;
    std::set<const Node*> m_nodes_with_attributes;
};
