//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory>

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            enum class OP_TYPEID;
            class NodeWrapper;
        }
    }
}

#define NGRAPH_OP_LIST(a) a##_TYPEID,
enum class ngraph::runtime::interpreter::OP_TYPEID
{
#include "op.tbl"
};
#undef NGRAPH_OP_LIST

class ngraph::runtime::interpreter::NodeWrapper
{
public:
    NodeWrapper(const std::shared_ptr<ngraph::Node>& node,
                ngraph::runtime::interpreter::OP_TYPEID tid)
        : m_node{node}
        , m_typeid{tid}
    {
    }

    Node& get_node() const { return *m_node; }
    ngraph::runtime::interpreter::OP_TYPEID get_typeid() const { return m_typeid; }
private:
    std::shared_ptr<ngraph::Node> m_node;
    OP_TYPEID m_typeid;
};
