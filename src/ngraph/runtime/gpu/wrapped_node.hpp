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

#include <memory>
#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/gpu/emitters/softmax.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include <type_traits>

namespace ngraph
{
    namespace op
    {
        namespace gpu
        {
            template <typename NODE_TYPE>
            class MemoryWrappedNode : public NODE_TYPE
            {
            public:
                template <typename... Args>
                MemoryWrappedNode(Args&&... args)
                    : NODE_TYPE(std::forward<Args>(args)...)
                    , m_emitter(this)
                {
                    // add constant memory input
                    size_t i = this->m_inputs.size();
                    for (auto& data : m_emitter.get_constants())
                    {
                        auto constant = std::make_shared<op::Constant>(
                            ngraph::element::from<typename std::remove_reference<decltype(data[0])>::type>(),
                            Shape{data.size()}, data);
                        this->m_inputs.emplace_back(
                            this, i++, constant->get_outputs().at(0));
                    }

                    // add worskapce output
                    for (auto& workspace : m_emitter.get_workspaces())
                    {
                        this->add_output(this->get_element_type(), workspace);
                    }
                }

                MemoryWrappedNode(const NODE_TYPE& node)
                    : NODE_TYPE(node)
                    , m_emitter(this)
                {
                    // add constant memory input
                    size_t i = this->m_inputs.size();
                    for (auto& data : m_emitter.get_constants())
                    {
                        auto constant = std::make_shared<op::Constant>(
                            ngraph::element::from<typename std::remove_reference<decltype(data[0])>::type>(),
                            Shape{data.size()}, data);
                        this->m_inputs.emplace_back(
                            this, i++, constant->get_outputs().at(0));
                    }

                    // add worskapce output
                    for (auto& workspace : m_emitter.get_workspaces())
                    {
                        this->add_output(this->get_element_type(), workspace);
                    }
                }

            private:
                runtime::gpu::Emitter<NODE_TYPE> m_emitter;
            };

            template <typename NODE_TYPE, typename... Args>
            std::shared_ptr<Node> make_wrapped_node(Args&&... args)
            {
                return std::make_shared<MemoryWrappedNode<NODE_TYPE>>(std::forward<Args>(args)...);
            }

            template <typename NODE_TYPE>
            std::shared_ptr<Node> make_wrapped_node(const NODE_TYPE& node)
            {
                return std::make_shared<MemoryWrappedNode<NODE_TYPE>>(node);
            }
        }
    }
}
