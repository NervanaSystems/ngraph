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

                template <template <typename NODE_TYPE> class CONTAINER_NODE_TYPE,
                          typename NODE_TYPE>
                bool package_node(std::shared_ptr<ngraph::Node> node);
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

template <template <class NODE_TYPE> class CONTAINER_NODE_TYPE, class NODE_TYPE>
bool ngraph::runtime::gpu::pass::package_node(std::shared_ptr<ngraph::Node> node)
{
    std::cout << node->get_name() << " before wrap" << std::endl;
    if (auto native_node = std::dynamic_pointer_cast<NODE_TYPE>(node))
    {
        if (std::dynamic_pointer_cast<ngraph::op::gpu::RequiresEmitter>(node) != nullptr)
        {
            return false;
        }

        auto container_node = std::make_shared<CONTAINER_NODE_TYPE<NODE_TYPE>>(native_node);

        container_node->graph_replace();

        return true;
    }
    return false;
}
