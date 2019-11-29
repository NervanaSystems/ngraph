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

#include <set>

#include "ngraph/node.hpp"

namespace ngraph
{
    class OpSet
    {
    public:
        OpSet(const std::set<NodeTypeInfo>& op_types)
            : m_op_types(op_types)
        {
        }

        template <typename T>
        bool contains_type() const
        {
            return m_op_types.find(T::type_info) != m_op_types.end();
        }

        bool contains_op_type(Node* node) const
        {
            return m_op_types.find(node->get_type_info()) != m_op_types.end();
        }

    protected:
        std::set<NodeTypeInfo> m_op_types;
    };

    const OpSet& get_opset0();
    const OpSet& get_opset1();
}