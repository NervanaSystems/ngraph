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

#include <sstream>

#include "common_function_collection.hpp"

using namespace std;
using namespace ngraph;

pass::CommonFunctionCollection::CommonFunctionCollection(function<string(Node&, string)> emitter,
                                                         unordered_map<Node*, Node*>& result_map,
                                                         string& emitted_functions)
    : m_emit_op_as_function(emitter)
    , m_node_function_map(result_map)
    , m_emitted_functions(emitted_functions)
{
}

pass::CommonFunctionCollection::~CommonFunctionCollection()
{
}

bool pass::CommonFunctionCollection::run_on_module(vector<shared_ptr<Function>>& functions)
{
    // This for loop creates a collection of functions that are called more than once
    // and emitting them as globally callable functions.
    unordered_map<string, Node*> match_function_map;
    stringstream ss;
    const string function_name = "__f__";
    for (const shared_ptr<Function>& current_function : functions)
    {
        list<shared_ptr<Node>> op_list = current_function->get_ordered_ops();
        if (op_list.size() < 2)
        {
            // Since we are comparing ops there must be at least two ops to proceed.
            continue;
        }
        for (const shared_ptr<Node>& op : op_list)
        {
            if (op->is_constant() || op->is_parameter())
            {
                continue;
            }

            Node& node = *op;
            string match_function = m_emit_op_as_function(node, function_name);
            auto it = match_function_map.find(match_function);
            if (it != match_function_map.end())
            {
                m_node_function_map.insert({&node, it->second});
                m_node_function_map.insert({it->second, it->second});
            }
            else
            {
                match_function_map.insert({match_function, &node});

                auto offset = match_function.find(function_name);
                string emitted_function = match_function;
                string match_function_name = create_function_name(node);
                emitted_function.replace(offset, 5, match_function_name);
                ss << emitted_function << "\n";
            }
        }
    }
    m_emitted_functions = ss.str();
    return false;
}

string pass::CommonFunctionCollection::create_function_name(const Node& node)
{
    return "func_" + node.get_name();
}
