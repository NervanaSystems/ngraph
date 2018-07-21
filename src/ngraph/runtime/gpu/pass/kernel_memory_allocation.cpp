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

#include <memory>
#include <unordered_map>

#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/gpu/emitters/softmax.hpp"

#include "ngraph/runtime/gpu/pass/kernel_memory_allocation.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/gpu/wrapped_node.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool replace_softmax(std::shared_ptr<Node> node)
{
    if (auto softmax = std::dynamic_pointer_cast<op::Softmax>(node))
    {
        if (auto prev_wrapped = std::dynamic_pointer_cast<op::gpu::MemoryWrappedNode<op::Softmax>>(node))
        {
            return false;
        }

        auto wrapped = std::make_shared<ngraph::op::gpu::MemoryWrappedNode<ngraph::op::Softmax>>(softmax);
        auto softmax_output = std::make_shared<op::GetOutputElement>(wrapped, 0);
        for (auto& user : node->get_users())
        {
            for (size_t i = 0; i < user->get_input_size(); i++)
            {
                if (user->get_argument(i) == node)
                {
                    user->get_inputs().at(i).replace_output(softmax_output->get_outputs().at(0));
                }
            }

        }

        return true;
    }

    return false;
}

static std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>
    initialize_ops_to_replace()
{
    return std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>(
        {{TI(op::Softmax), replace_softmax}});
}

static std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>
    ops_to_replace = initialize_ops_to_replace();

bool ngraph::runtime::gpu::pass::KernelMemoryAllocation::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        const Node& node = *n;
        auto it = ops_to_replace.find(TI(node));
        if (it == ops_to_replace.end())
        {
            continue;
        }

        replaced = it->second(n) || replaced;
    }
    return replaced;
}
