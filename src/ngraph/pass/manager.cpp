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
#ifdef WIN32
#else
#include <cxxabi.h>
#endif
#include <iomanip>
#include <iostream>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/serialize.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

ngraph::pass::Manager::Manager()
{
    static const auto nevt = std::getenv("NGRAPH_ENABLE_VISUALIZE_TRACING");
    if (nevt)
    {
        m_visualize = true;
    }
    static const auto nest = std::getenv("NGRAPH_ENABLE_SERIALIZE_TRACING");
    if (nest)
    {
        m_serialize = true;
    }
}

ngraph::pass::Manager::~Manager()
{
}

void ngraph::pass::Manager::initialize_default_passes()
{
}

void ngraph::pass::Manager::run_passes(shared_ptr<Function> func, bool transitive)
{
    bool profile_enabled = getenv("NGRAPH_PROFILE_PASS_ENABLE") != nullptr;

    vector<shared_ptr<Function>> fs;
    if (transitive)
    {
        // find all functions
        traverse_functions(func, [&](shared_ptr<Function> f) { fs.push_back(f); });
    }
    else
    {
        fs = {func};
    }
    set<shared_ptr<Function>> tfs(begin(fs), end(fs));
    get_state().set_functions(tfs);

    size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    for (shared_ptr<PassBase> pass : m_pass_list)
    {
        pass_timer.start();
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

        if (m_visualize || m_serialize)
        {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = fs.at(0)->get_name() + std::string("_") + index_str +
                                 std::string("_") + m_pass_names.at(index) + std::string(".");

            if (m_visualize)
            {
                pass::VisualizeTree vt(base_filename + pass::VisualizeTree::get_file_ext());
                vt.run_on_module(fs);
            }

            if (m_serialize)
            {
                // no "." in the extension
                pass::Serialization st(base_filename + "json");
                st.run_on_module(fs);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled)
        {
            PassBase* p = pass.get();
            string name = typeid(*p).name();
#ifndef WIN32
            int status;
            name = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
#endif
            cout << setw(7) << pass_timer.get_milliseconds() << "ms " << name << "\n";
        }
    }
    if (profile_enabled)
    {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
}

ngraph::pass::ManagerState& ngraph::pass::Manager::get_state()
{
    return m_state;
}
