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

#include "ngraph/runtime/interpreter/node_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::interpreter::NodeWrapper::NodeWrapper(const shared_ptr<const Node>& node)
    : m_node{node}
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {"Abs", runtime::interpreter::OP_TYPEID::Abs},
// {"Acos", runtime::interpreter::OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a) {#a, runtime::interpreter::OP_TYPEID::a},
    static unordered_map<string, runtime::interpreter::OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
    };
#undef NGRAPH_OP

    auto it = typeid_map.find(m_node->description());
    if (it != typeid_map.end())
    {
        m_typeid = it->second;
    }
    else
    {
        throw unsupported_op("Unsupported op '" + m_node->description() + "'");
    }
}
