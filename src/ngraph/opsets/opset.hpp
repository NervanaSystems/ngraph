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

#include <map>
#include <mutex>
#include <set>

#include "ngraph/factory.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    class OpSet
    {
        static std::mutex& get_mutex();

    public:
        OpSet() = default;

        template <typename T>
        void insert()
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            m_op_types.insert(T::type_info);
            m_name_type_info_map[T::type_info.name] = T::type_info;
            ngraph::FactoryRegistry<Node>::get().register_factory<T>();
        }

        ngraph::Node* create(const std::string& name);

        template <typename T>
        bool contains_type() const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(T::type_info) != m_op_types.end();
        }

        bool contains_op_type(Node* node) const
        {
            std::lock_guard<std::mutex> guard(get_mutex());
            return m_op_types.find(node->get_type_info()) != m_op_types.end();
        }

    protected:
        std::set<NodeTypeInfo> m_op_types;
        std::map<std::string, NodeTypeInfo> m_name_type_info_map;
    };

    const OpSet& get_opset0();
    const OpSet& get_opset1();
}