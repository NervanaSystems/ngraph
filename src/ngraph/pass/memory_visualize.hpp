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

#include <iostream>
#include <limits>
#include <list>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class MemoryVisualize;
    }
}

class ngraph::pass::MemoryVisualize : public ModulePass
{
public:
    MemoryVisualize(const std::string& filename);
    virtual bool run_on_module(std::vector<Function*>&) override;

private:
    const Node* find_largest_op(const std::list<Node*>& nodes);
    void draw_tensor_weight(std::ostream& file, const std::list<Node*>& nodes);
    void draw_histogram(std::ostream& file, const std::list<Node*>& nodes);
    void draw_op_influence(std::ostream& file, const std::list<Node*>& nodes);
    int compute_op_weight(const Node* exop);

    static size_t memory_usage(const Node*);
    static size_t memory_footprint(const Node*);
    static size_t memory_footprint(const std::list<Node*>&);

    const std::string m_filename;
};
