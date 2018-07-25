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

#include <algorithm>
#include <memory>
#include <type_traits>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/runtime/gpu/op/emittable_node.hpp"

namespace ngraph
{
    namespace op
    {
        namespace gpu
        {
            template <typename NODE_TYPE>
            class MemoryWrappedNode : public EmittableNode<NODE_TYPE>
            {
            public:
                MemoryWrappedNode(const std::shared_ptr<NODE_TYPE>& node)
                    : EmittableNode<NODE_TYPE>(node)
                {
                    add_inputs();
                    add_outputs();
                }

                MemoryWrappedNode(const std::shared_ptr<NODE_TYPE>& node, const NodeVector& args)
                    : EmittableNode<NODE_TYPE>(node, args)
                {
                    add_outputs();
                }

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& args) const override
                {
                    // clone underlying native node with new args
                    NodeVector new_args;
                    std::copy(args.begin(),
                              args.begin() + this->m_node->get_arguments().size(),
                              std::back_inserter(new_args));
                    auto new_node =
                        std::dynamic_pointer_cast<NODE_TYPE>(this->m_node->copy_with_new_args(new_args));

                    // construct new wrapped node passing the same native inputs and wrapped constants
                    return std::make_shared<MemoryWrappedNode<NODE_TYPE>>(new_node, args);
                }

            protected:
                void add_inputs()
                {
                    // add constant memory input
                    size_t i = this->m_inputs.size();
                    for (auto& data : this->m_emitter.get_constants())
                    {
                        auto constant = std::make_shared<op::Constant>(
                            ngraph::element::from<
                                typename std::remove_reference<decltype(data[0])>::type>(),
                            Shape{data.size()},
                            data);
                        this->m_inputs.emplace_back(this, i++, constant->get_outputs().at(0));
                    }
                }
                void add_outputs()
                {
                    // add node's outputs to wrapped node
                    size_t i = 0;
                    for (auto& output : this->m_node->get_outputs())
                    {
                        this->m_outputs.emplace_back(this, i++, output.get_tensor_view());
                    }

                    // add worskapce output
                    for (auto& workspace : this->m_emitter.get_workspaces())
                    {
                        this->add_output(this->m_node->get_element_type(), workspace);
                    }
                }
            };
        }
    }
}
