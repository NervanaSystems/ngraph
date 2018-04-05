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
#include "ngraph/pass/visualize_tree.hpp"

using namespace std;
using namespace ngraph;

ngraph::pass::Manager::Manager()
{
    static const auto nevt = std::getenv("NGRAPH_ENABLE_VISUALIZE_TRACING");
    if (nevt)
    {
        m_visualize = true;
    }
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

    size_t index = 0;
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

        if (m_visualize)
        {
            //visualizations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto fname = fs.at(0)->get_name() + std::string("_") + index_str + std::string("_") +
                         m_pass_names.at(index) + std::string(".") +
                         pass::VisualizeTree::get_file_ext();
            pass::VisualizeTree vt(fname);
            vt.run_on_module(fs);
        }
        index++;
    }
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
