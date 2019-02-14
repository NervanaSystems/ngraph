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

#include "ngraph/runtime/hybrid/hybrid_executable.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/runtime/hybrid/pass/default_placement.hpp"
#include "ngraph/runtime/hybrid/pass/dump.hpp"
#include "ngraph/runtime/hybrid/pass/fix_get_output_element.hpp"
#include "ngraph/runtime/hybrid/pass/liveness.hpp"
#include "ngraph/runtime/hybrid/pass/memory_layout.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace ngraph;
using namespace std;

static void node_modifiers(const Node& node, vector<string>& attributes)
{
    vector<string> colors = {"\"#A0FFA0\"", "\"#FFF790\""};
    if (node.get_placement_index() < colors.size())
    {
        string color = colors[node.get_placement_index()];
        attributes.push_back("style=filled");
        attributes.push_back("fillcolor=" + color);
    }
}

runtime::hybrid::HybridExecutable::HybridExecutable(
    const std::vector<std::shared_ptr<runtime::Backend>>& backend_list,
    const shared_ptr<Function>& func,
    bool enable_performance_collection,
    bool debug_enabled)
    : m_function{func}
    , m_backend_list{backend_list}
    , m_debug_enabled{debug_enabled}
{
    {
        // Run placement pass
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<runtime::hybrid::pass::DefaultPlacement>(m_backend_list);
        pass_manager.register_pass<runtime::hybrid::pass::FixGetOutputElement>();
        pass_manager.register_pass<runtime::hybrid::pass::Liveness>();
        pass_manager.register_pass<runtime::hybrid::pass::Dump>("graph.dump");
        // pass_manager.register_pass<runtime::hybrid::pass::MemoryLayout>();
        if (m_debug_enabled)
        {
            pass_manager.register_pass<ngraph::pass::VisualizeTree>("graph.png", node_modifiers);
        }
        pass_manager.run_passes(m_function);

        // Split function to sub_functions
        tie(m_sub_functions, m_map_parameter_to_result) =
            runtime::hybrid::split_function_by_placement(m_function);

        // Compile subfunctions in corresponding backends
        size_t subfunction_number = 0;
        for (shared_ptr<Function>& sub_function : m_sub_functions)
        {
            size_t placement = sub_function->get_placement();
            if (m_debug_enabled)
            {
                string name = "subfunction_" + to_string(subfunction_number++);
                ngraph::pass::Manager pm;
                pm.register_pass<ngraph::pass::VisualizeTree>(name + ".png", node_modifiers);
                pm.register_pass<runtime::hybrid::pass::Dump>(name + ".dump");
                pm.run_passes(sub_function);
            }
            auto backend = m_backend_list[placement];
            shared_ptr<Executable> exec = backend->compile(sub_function);
            m_executable_map[sub_function] = exec;

            // Compile will replace nodes so we need to make one more pass through all
            // ops to reset placement
            for (auto op : sub_function->get_ops())
            {
                op->set_placement_index(placement);
            }
        }
    }

    set_parameters_and_results(*func);
}

bool runtime::hybrid::HybridExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                             const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    bool rc = true;

    using node_map_t = unordered_map<shared_ptr<Node>, shared_ptr<runtime::Tensor>>;

    // Parameter and result node in sub_function maps to one Tensor
    node_map_t map_node_to_tensor;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        map_node_to_tensor[m_function->get_parameters()[i]] = inputs[i];
    }
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        map_node_to_tensor[m_function->get_results()[i]] = outputs[i];
    }

    // Call subfunctions
    for (const shared_ptr<Function>& sub_function : m_sub_functions)
    {
        // Init backend
        size_t placement = sub_function->get_placement();
        auto backend = m_backend_list[placement];

        // Prepare parameter Tensors
        vector<shared_ptr<runtime::Tensor>> parameters;
        for (const shared_ptr<op::Parameter>& parameter_node : sub_function->get_parameters())
        {
            auto it = map_node_to_tensor.find(parameter_node);
            if (it != map_node_to_tensor.end())
            {
                if (it->second->get_parent() == backend.get())
                {
                    parameters.push_back(it->second);
                }
                else
                {
                    auto parameter = backend->create_tensor(parameter_node->get_element_type(),
                                                            parameter_node->get_shape());
                    parameter->copy_from(*(it->second));
                    parameters.push_back(parameter);
                }
            }
            else
            {
                // Handle temporary tensors that go between subgraphs
                auto result_node = m_map_parameter_to_result.at(parameter_node);
                auto result = map_node_to_tensor.at(result_node);
                auto parameter = backend->create_tensor(parameter_node->get_element_type(),
                                                        parameter_node->get_shape());
                parameter->copy_from(*result);
                map_node_to_tensor[parameter_node] = parameter;
                parameters.push_back(parameter);
            }
        }

        // Prepare result Tensors
        vector<shared_ptr<runtime::Tensor>> results;
        map<runtime::Tensor*, runtime::Tensor*> copy_back;
        for (const shared_ptr<op::Result>& result_node : sub_function->get_results())
        {
            auto it = map_node_to_tensor.find(result_node);
            if (it != map_node_to_tensor.end())
            {
                if (it->second->get_parent() == backend.get())
                {
                    results.push_back(it->second);
                }
                else
                {
                    auto result = backend->create_tensor(result_node->get_element_type(),
                                                         result_node->get_shape());
                    results.push_back(result);
                    copy_back.insert({result.get(), it->second.get()});
                }
            }
            else
            {
                // Handle temporary tensors that go between subgraphs
                auto result = backend->create_tensor(result_node->get_element_type(),
                                                     result_node->get_shape());
                map_node_to_tensor[result_node] = result;
                results.push_back(result);
            }
        }

        // Call
        auto exec = m_executable_map[sub_function];
        exec->call(results, parameters);

        // Need to copy any results to the correct device
        for (const auto& p : copy_back)
        {
            p.second->copy_from(*p.first);
        }
    }
    return rc;
}

size_t runtime::hybrid::HybridExecutable::get_placement(const runtime::Tensor* t)
{
    size_t index = 0;
    for (const shared_ptr<ngraph::runtime::Backend>& be : m_backend_list)
    {
        if (t->get_parent() == be.get())
        {
            return index;
        }
        index++;
    }
    return -1;
}
