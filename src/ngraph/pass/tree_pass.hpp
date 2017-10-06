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

#include <list>
#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class TreeBase;
    }
}

class ngraph::pass::TreeBase : public Base
{
public:
    virtual ~TreeBase() {}
    // return true if changes were made to the tree
    virtual bool run_on_tree(std::shared_ptr<Node>) = 0;

    // derived class throws exception if its dependencies have not been met
    virtual void check_dependencies(const std::vector<std::shared_ptr<TreeBase>>&) const {}
};
