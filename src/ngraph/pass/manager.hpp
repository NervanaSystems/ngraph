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

#include <vector>

#include "call_pass.hpp"
#include "tree_pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class Manager;
    }

    class Node;
}

class ngraph::pass::Manager
{
public:
    Manager();
    ~Manager();

    void initialize_default_passes();

    void register_pass(std::shared_ptr<TreeBase>);
    void register_pass(std::shared_ptr<CallBase>);

    void run_passes(std::shared_ptr<Node> nodes);

    const std::list<Node*>& get_sorted_list() const;

private:
    std::vector<std::shared_ptr<TreeBase>>  m_tree_passes;
    std::vector<std::shared_ptr<CallBase>>  m_call_passes;
    std::list<Node*>                        m_sorted_list;
};
