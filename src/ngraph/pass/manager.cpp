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
#include "ngraph/pass/pass.hpp"

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

void ngraph::pass::Manager::run_passes(shared_ptr<Function> func)
{
    run_passes(func.get());
}

void ngraph::pass::Manager::run_passes(Function* func)
{
    vector<Function*> fs = {func};
    get_state().set_functions(fs);

    for (shared_ptr<PassBase> pass : m_pass_list)
    {
        auto module_pass = dynamic_pointer_cast<ModulePass>(pass);
        auto function_pass = dynamic_pointer_cast<FunctionPass>(pass);
        auto node_pass = dynamic_pointer_cast<NodePass>(pass);
        auto call_graph_pass = dynamic_pointer_cast<CallGraphPass>(pass);
        if (module_pass)
        {
            NGRAPH_INFO;
            module_pass->run_on_module(fs);
            NGRAPH_INFO;
        }
        else if (function_pass)
        {
            NGRAPH_INFO;
            for (Function* f : fs)
            {
                NGRAPH_INFO;
                function_pass->run_on_function(f);
                NGRAPH_INFO;
            }
            NGRAPH_INFO;
        }
        else if (node_pass)
        {
            NGRAPH_INFO;
            for (Function* f : fs)
            {
                NGRAPH_INFO;
                for (Node* n : f->get_ops())
                {
                    NGRAPH_INFO;
                    node_pass->run_on_node(n);
                    NGRAPH_INFO;
                }
                NGRAPH_INFO;
            }
            NGRAPH_INFO;
        }
        else if (call_graph_pass)
        {
            NGRAPH_INFO;
            for (Function* f : fs)
            {
                NGRAPH_INFO;
                call_graph_pass->run_on_call_graph(f->get_ordered_ops());
                NGRAPH_INFO;
            }
            NGRAPH_INFO;
        }
    }
    // for (shared_ptr<ModulePass>& p : m_module_passes)
    // {
    //     p->set_state(get_state());
    //     p->run_on_module(fs);
    // }

    // for (Function* f : fs)
    // {
    //     for (shared_ptr<FunctionPass> p : m_function_passes)
    //     {
    //         p->set_state(get_state());
    //         p->run_on_function(f);
    //     }
    // }

    // for (Function* f : fs)
    // {
    //     NGRAPH_INFO;
    //     for (shared_ptr<NodePass> p : m_node_passes)
    //     {
    //         for (Node* node : f->get_ops())
    //         {
    //             NGRAPH_INFO;
    //             p->set_state(get_state());
    //             p->run_on_node(node);
    //         }
    //     }
    // }

    // for (shared_ptr<CallGraphPass>& p : m_call_graph_passes)
    // {
    //     p->set_state(get_state());
    //     p->run_on_call_graph(func->get_ordered_ops());
    // }
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
