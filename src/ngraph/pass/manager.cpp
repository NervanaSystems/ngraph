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

#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager.hpp"

using namespace std;
using namespace ngraph;

ngraph::pass::Manager::Manager()
{
}

ngraph::pass::Manager::~Manager()
{
}

void ngraph::pass::Manager::initialize_default_passes()
{
}

void ngraph::pass::Manager::register_pass_ptr(std::shared_ptr<CallBase> p)
{
    if (p == nullptr)
    {
        throw invalid_argument("null pass registered");
    }
    p->check_dependencies(m_call_passes);
    m_call_passes.push_back(p);
}

void ngraph::pass::Manager::register_pass_ptr(std::shared_ptr<FunctionPass> p)
{
    if (p == nullptr)
    {
        throw invalid_argument("null pass registered");
    }
    p->check_dependencies(m_function_passes);
    m_function_passes.push_back(p);
}

void ngraph::pass::Manager::run_passes(shared_ptr<Function> func)
{
    run_passes(func.get());
}

void ngraph::pass::Manager::run_passes(Function* func)
{
    for (shared_ptr<FunctionPass> p : m_function_passes)
    {
        p->set_state(get_state());
        p->run_on_function(func);
    }

    for (shared_ptr<CallBase>& p : m_call_passes)
    {
        p->set_state(get_state());
        p->run_on_call_list(func->get_ordered_ops());
    }
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
