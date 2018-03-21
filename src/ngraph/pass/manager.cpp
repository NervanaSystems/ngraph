/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <iostream>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"

using namespace std;
using namespace ngraph;

ngraph::pass::Manager::Manager()
{
}

ngraph::pass::Manager::Manager(bool to_set_is_output)
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
    // find all functions
    vector<shared_ptr<Function>> fs;
    traverse_functions(func, [&](shared_ptr<Function> f) { fs.push_back(f); });

    set<shared_ptr<Function>> tfs(begin(fs), end(fs));
    get_state().set_functions(tfs);

    for (shared_ptr<PassBase> pass : m_pass_list)
    {
        pass->set_state(get_state());
        auto module_pass = dynamic_pointer_cast<ModulePass>(pass);
        auto function_pass = dynamic_pointer_cast<FunctionPass>(pass);
        auto node_pass = dynamic_pointer_cast<NodePass>(pass);
        auto call_graph_pass = dynamic_pointer_cast<CallGraphPass>(pass);
        if (module_pass)
        {
            module_pass->run_on_module(fs);
        }
        else if (function_pass)
        {
            for (shared_ptr<Function> f : fs)
            {
                function_pass->run_on_function(f);
            }
        }
        else if (node_pass)
        {
            for (shared_ptr<Function> f : fs)
            {
                for (shared_ptr<Node> n : f->get_ops())
                {
                    node_pass->run_on_node(n);
                }
            }
        }
        else if (call_graph_pass)
        {
            for (shared_ptr<Function> f : fs)
            {
                call_graph_pass->run_on_call_graph(f->get_ordered_ops());
            }
        }
    }
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
