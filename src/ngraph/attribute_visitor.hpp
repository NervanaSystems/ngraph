//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <unordered_map>
#include <utility>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    template <typename T>
    class ValueAccessor;
    class VisitorAdapter;
    class Node;

    /// \brief Visits the attributes of a node.
    ///
    /// Attributes are the values set when building a graph which are not
    /// computed as the graph executes. Values computed from the graph topology and attributes
    /// during compilation are not attributes.
    ///
    /// Attributes can have a wide variety of types. In order for a visitor to handle a wide
    /// range of types, some defined in other modules. on_attribute for string and bool attributes
    /// must be implemented directly by a visitor. For other attribute types, an attribute visitor
    /// uses an AttributeAdapter<T> class to convert a T& to a ValueAccessor<T>& derived from
    /// ValueAccessor<void>. If the attribute visitor overrides one of the optional on_adapter
    /// methods, that method will be called; otherwise the default implementation for
    /// on_adapter will call the on_adapter method for ValueAccessor<void>&.
    ///
    /// Why are there optional on_adapter methods? This allows new on_adapter methods to be added
    /// without requiring AttributeVisitors to be immediately updated.
    ///
    /// Why aren't all the methods on_attribute? Either there was some template issue related to the
    /// generic on_attribute, or it was related to preventing API changes. similarly the string and
    /// bool on_attributes are related to API stability.
    class NGRAPH_API AttributeVisitor
    {
    public:
        enum class ContextType
        {
            Struct,
            Vector
        };
        struct Context
        {
            ContextType context_type;
            std::string name;
            int64_t index;
        };

        virtual ~AttributeVisitor() {}
        // Must implement these methods
        /// \brief handles all specialized on_adapter methods implemented by the visitor.
        ///
        /// The adapter implements get_type_info(), which can be used to determine the adapter
        /// directly
        /// or via is_type and as_type on any platform
        virtual void on_adapter(const std::string& name, ValueAccessor<void>& adapter) = 0;
        // The remaining adapter methods fall back on the void adapter if not implemented
        virtual void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<bool>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<int8_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<int16_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<int32_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<uint8_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<uint16_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<uint32_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<uint64_t>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<float>& adapter);
        virtual void on_adapter(const std::string& name, ValueAccessor<double>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int8_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int16_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int32_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int64_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint8_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint16_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint32_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint64_t>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<float>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<double>>& adapter);
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<std::string>>& adapter);
        /// \brief Hook for adapters that need visitor access
        virtual void on_adapter(const std::string& name, VisitorAdapter& adapter);

        /// Legacy method
        virtual void on_attribute(const std::string& name, std::string& value)
            NGRAPH_DEPRECATED("Override on_adapter instead");
        /// Legacy method
        virtual void on_attribute(const std::string& name, bool& value)
            NGRAPH_DEPRECATED("Override on_adapter instead");
        /// The generic visitor. There must be a definition of AttributeAdapter<T> that can convert
        /// to a ValueAccessor<U> for one of the on_adpater methods.
        template <typename AT>
        void on_attribute(const std::string& name, AT& value)
        {
            AttributeAdapter<AT> adapter(value);
            on_adapter(name, adapter);
        }
        /// \returns The nested context of visits
        const std::vector<Context>& get_context() const { return m_context; }
        /// \returns context prepended to names
        virtual std::string get_name_with_context(const std::string& name);
        /// \brief Start visiting a nested structure
        virtual void start_structure(const std::string& name);
        /// \brief Finish visiting a nested structure
        virtual void finish_structure();
        virtual void start_vector(const std::string& name);
        virtual void next_vector_element();
        virtual void finish_vector();
        /// \brief Associate a node with an id.
        ///
        /// No node may be used as an attribute unless it has already been registered with an ID.
        /// References to nodes are visited with a ValueAccessor of their ID.
        virtual void register_node(const std::shared_ptr<Node>& node, int64_t id);
        /// Returns the node with the given id, or nullptr if there is no registered node
        virtual std::shared_ptr<Node> get_registered_node(int64_t id);
        /// Returns the id for the node, or -1 if the node is not registered
        virtual int64_t get_registered_node_id(const std::shared_ptr<Node>& node);

    protected:
        std::vector<Context> m_context;
        std::unordered_map<std::shared_ptr<Node>, int64_t> m_node_id_map;
        std::unordered_map<int64_t, std::shared_ptr<Node>> m_id_node_map;
    };
}
