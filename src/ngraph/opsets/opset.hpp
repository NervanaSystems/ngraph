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

#include <map>
#include <mutex>
#include <set>

#include "ngraph/factory.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    /// \brief Run-time opset information
    class NGRAPH_API OpSet
    {
        static std::mutex& get_mutex();

    public:
        OpSet() {}
        std::set<NodeTypeInfo>::size_type size() const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.size();
        }
        /// \brief Insert an op into the opset with a particular name and factory
        void insert(const std::string& name,
                    const NodeTypeInfo& type_info,
                    FactoryRegistry<Node>::Factory factory)
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            m_op_types.insert(type_info);
            m_name_type_info_map[name] = type_info;
            m_factory_registry.register_factory(type_info, factory);
        }

        /// \brief Insert OP_TYPE into the opset with a special name and the default factory
        template <typename OP_TYPE>
        void insert(const std::string& name)
        {
            insert(name, OP_TYPE::type_info, FactoryRegistry<Node>::get_default_factory<OP_TYPE>());
        }

        /// \brief Insert OP_TYPE into the opset with the default name and factory
        template <typename OP_TYPE>
        void insert()
        {
            insert<OP_TYPE>(OP_TYPE::type_info.name);
        }

        const std::set<NodeTypeInfo>& get_types_info() const { return m_op_types; }
        /// \brief Create the op named name using it's factory
        ngraph::Node* create(const std::string& name) const;

        /// \brief Return true if OP_TYPE is in the opset
        bool contains_type(const NodeTypeInfo& type_info) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(type_info) != m_op_types.end();
        }

        /// \brief Return true if OP_TYPE is in the opset
        template <typename OP_TYPE>
        bool contains_type() const
        {
            return contains_type(OP_TYPE::type_info);
        }

        /// \brief Return true if name is in the opset
        bool contains_type(const std::string& name) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_name_type_info_map.find(name) != m_name_type_info_map.end();
        }

        /// \brief Return true if node's type is in the opset
        bool contains_op_type(Node* node) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(node->get_type_info()) != m_op_types.end();
        }

        ngraph::FactoryRegistry<ngraph::Node>& get_factory_registry() { return m_factory_registry; }
    protected:
        ngraph::FactoryRegistry<ngraph::Node> m_factory_registry;
        std::set<NodeTypeInfo> m_op_types;
        std::map<std::string, NodeTypeInfo> m_name_type_info_map;
    };

    const NGRAPH_API OpSet& get_opset0();
    const NGRAPH_API OpSet& get_opset1();
    const NGRAPH_API OpSet& get_opset2();
}
