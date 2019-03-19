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

    /// \brief A handle for one of a node's inputs.
    class NodeInput
    {
    public:
        /// \brief Constructs a NodeInput.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        NodeInput(Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        Node* get_node() const { return m_node; }
        /// \return The index of the input referred to by this input handle.
        size_t get_index() const { return m_index; }
        /// \return The element type of the input referred to by this input handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_input_element_type(m_index);
        }
        /// \return The shape of the input referred to by this input handle.
        const Shape& get_shape() const { return m_node->get_input_shape(m_index); }
        /// \return The partial shape of the input referred to by this input handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_input_partial_shape(m_index);
        }
        /// \return A handle to the output that is connected to this input.
        NodeOutput get_source_output() const;

        bool operator==(const NodeInput& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const NodeInput& other) const
        {
            return m_node != other.m_node || m_index != other.m_index;
        }
        bool operator<(const NodeInput& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const NodeInput& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const NodeInput& other) const
        {
            return m_node <= other.m_node || (m_node == other.m_node && m_index <= other.m_index);
        }
        bool operator>=(const NodeInput& other) const
        {
            return m_node >= other.m_node || (m_node == other.m_node && m_index >= other.m_index);
        }

    private:
        Node* const m_node;
        const size_t m_index;
    };
} // namespace ngraph
