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

#include <iostream>
#include <memory>

#include "ngraph/log.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/node.hpp"
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph;

Function* ngraph::pass::ManagerState::get_function()
{
    return m_function;
}

void ngraph::pass::ManagerState::set_function(Function* func)
{
    m_function = func;
}

size_t ngraph::pass::ManagerState::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void ngraph::pass::ManagerState::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}

std::list<Node*>& ngraph::pass::ManagerState::get_call_graph()
{
    return m_call_graph;
}

const std::list<Node*>& ngraph::pass::ManagerState::get_call_graph() const
{
    return m_call_graph;
}

ngraph::pass::Manager::Manager()
{
}

ngraph::pass::Manager::~Manager()
{
}

void ngraph::pass::Manager::initialize_default_passes()
{
}

void ngraph::pass::Manager::register_pass(std::shared_ptr<TreeBase> p)
{
    if (p == nullptr)
    {
        throw invalid_argument("null pass registered");
    }
    p->check_dependencies(m_tree_passes);
    m_tree_passes.push_back(p);
}

void ngraph::pass::Manager::register_pass(std::shared_ptr<CallBase> p)
{
    if (p == nullptr)
    {
        throw invalid_argument("null pass registered");
    }
    p->check_dependencies(m_call_passes);
    m_call_passes.push_back(p);
}

void ngraph::pass::Manager::run_passes(shared_ptr<Function> func)
{
    run_passes(func.get());
}

void ngraph::pass::Manager::run_passes(Function* func)
{
    m_state.set_function(func);
    for (shared_ptr<TreeBase> p : m_tree_passes)
    {
        p->set_state(get_state());
        p->run_on_tree(func->get_result());
    }

    for (shared_ptr<CallBase>& p : m_call_passes)
    {
        p->set_state(get_state());
        p->run_on_call_list(get_state().get_call_graph());
    }
}

const std::list<ngraph::Node*>& ngraph::pass::Manager::get_call_graph() const
{
    return m_state.get_call_graph();
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
