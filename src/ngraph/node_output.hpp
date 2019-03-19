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
    /// \brief A handle for one of a node's outputs.
    class NodeOutput
    {
    public:
        /// \brief Constructs a NodeOutput.
        /// \param node A `shared_ptr` to the node for the output handle.
        /// \param index The index of the output.
        NodeOutput(const std::shared_ptr<Node>& node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \brief Constructs a NodeOutput, referencing the zeroth output of the node.
        /// \param node A `shared_ptr` to the node for the output handle.
        template <typename T>
        NodeOutput(const std::shared_ptr<T>& node)
            : NodeOutput(node, 0)
        {
        }

        /// \return A `shared_ptr` to the node referred to by this output handle.
        const std::shared_ptr<Node>& get_node() const { return m_node; }
        /// \return The index of the output referred to by this output handle.
        size_t get_index() const { return m_index; }
        /// \return The element type of the output referred to by this output handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_output_element_type(m_index);
        }
        /// \return The shape of the output referred to by this output handle.
        const Shape& get_shape() const { return m_node->get_output_shape(m_index); }
        /// \return The partial shape of the output referred to by this output handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_output_partial_shape(m_index);
        }

        bool operator==(const NodeOutput& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const NodeOutput& other) const
        {
            return m_node != other.m_node || m_index != other.m_index;
        }
        bool operator<(const NodeOutput& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const NodeOutput& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const NodeOutput& other) const
        {
            return m_node <= other.m_node || (m_node == other.m_node && m_index <= other.m_index);
        }
        bool operator>=(const NodeOutput& other) const
        {
            return m_node >= other.m_node || (m_node == other.m_node && m_index >= other.m_index);
        }

    private:
        const std::shared_ptr<Node> m_node;
        const size_t m_index;
    };
} // namespace ngraph
