//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <list>
#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"

namespace ngraph
{
    namespace pass
    {
        class PassBase;
        class ModulePass;
        class FunctionPass;
        class NodePass;
        class CallGraphPass;
        class Manager;
    }
}

class ngraph::pass::PassBase
{
    friend class Manager;

public:
    virtual ~PassBase() {}
protected:
    ManagerState& get_state();
    void set_state(ManagerState&);

private:
    ManagerState* m_state;
};

class ngraph::pass::ModulePass : public PassBase
{
public:
    virtual ~ModulePass() {}
    virtual bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) = 0;
};

class ngraph::pass::FunctionPass : public PassBase
{
public:
    virtual ~FunctionPass() {}
    virtual bool run_on_function(std::shared_ptr<ngraph::Function>) = 0;
};

class ngraph::pass::NodePass : public PassBase
{
public:
    virtual ~NodePass() {}
    virtual bool run_on_node(std::shared_ptr<ngraph::Node>) = 0;
};

class ngraph::pass::CallGraphPass : public PassBase
{
public:
    virtual ~CallGraphPass() {}
    virtual bool run_on_call_graph(const std::list<std::shared_ptr<ngraph::Node>>&) = 0;
};
