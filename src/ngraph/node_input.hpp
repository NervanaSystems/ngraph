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

#pragma once

#include "ngraph/descriptor/input.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    class NodeOutput;

    class NodeInput
    {
    public:
        NodeInput(Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        NodeInput(std::shared_ptr<Node> n, size_t index)
            : NodeInput(n.get(), index)
        {
        }

        Node* get_node() const { return m_node; }
        size_t get_index() const { return m_index; }
        const element::Type& get_element_type() const { return m_node->get_input_element_type(m_index); }
        const Shape& get_shape() const { return m_node->get_input_shape(m_index); }
        const PartialShape& get_partial_shape() const { return m_node->get_input_partial_shape(m_index); }
        NodeOutput get_source_output() const;
    private:
        Node* const m_node;
        const size_t m_index;
    };
} // namespace ngraph
