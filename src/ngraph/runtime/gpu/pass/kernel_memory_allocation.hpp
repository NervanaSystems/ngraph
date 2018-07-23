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

#pragma once

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                class KernelMemoryAllocation;
            }
        }
    }
}

class ngraph::runtime::gpu::pass::KernelMemoryAllocation : public ngraph::pass::FunctionPass
{
public:
    KernelMemoryAllocation()
        : FunctionPass()
    {
    }

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);
};

template <typename NODE_TYPE>
bool export_kernel_memory_allocations(std::shared_ptr<ngraph::Node> node)
{
    if (auto original_node = std::dynamic_pointer_cast<NODE_TYPE>(node))
    {
        if (auto prev_wrapped =
            std::dynamic_pointer_cast<ngraph::op::gpu::MemoryWrappedNode_Base>(node))
        {
            return false;
        }

        // wrap node with user defined kernel allocations
        auto wrapped_node =
            std::make_shared<ngraph::op::gpu::MemoryWrappedNode<NODE_TYPE>>(original_node);

        // wire up native outputs of wrapped node to the graph
        for (size_t i = 0; i < original_node->get_outputs().size(); i++)
        {
            auto& node_output = original_node->get_outputs().at(i);
            std::set<ngraph::descriptor::Input*> copy_inputs{std::begin(node_output.get_inputs()),
                    std::end(node_output.get_inputs())};

            auto new_output = std::make_shared<ngraph::op::GetOutputElement>(wrapped_node, i);
            for (auto input : copy_inputs)
            {
                input->replace_output(new_output->get_outputs().at(0));
            }
        }
        return true;
    }
    return false;
}
