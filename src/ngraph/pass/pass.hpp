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

#include <list>
#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"
#include "ngraph/util.hpp"

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
        enum FusionType
        {
            //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
            // i.e. implement `generate_adjoints`
            DIFFERENTIABLE_FUSIONS = 0x1,
            REGULAR_FUSIONS = 0x2,
            ALL_FUSIONS = 0xFFFFFFFF
        };
    }
}

class ngraph::pass::PassBase
{
    enum class Property : uint32_t 
    {
        //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
        // i.e. implement `generate_adjoints`
        DIFFERENTIABLE_FUSIONS = 0x1,
        REGULAR_FUSIONS = 1 << 1,
        ALL_FUSIONS = REGULAR_FUSIONS | DIFFERENTIABLE_FUSIONS, 
        REQUIRE_STATIC_SHAPE = 1 << 2,
        CHANGE_FUNCTION_STATE = 1 << 3
    };
    friend class Manager;

public:
    virtual ~PassBase() {}
    bool get_property(const Property& prop);
protected:
    ManagerState& get_state();
    void set_state(ManagerState&);
    void set_property(const Property& prop, bool value);

private:
    EnumMask<Property> m_property;
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
