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

#include "ngraph/runtime/interpreter/node_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::interpreter::NodeWrapper::NodeWrapper(const shared_ptr<const Node>& node)
    : m_node{node}
{
    static const map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(a, b) {b::a::type_info, OP_TYPEID::a},
#include "ngraph/op/op_v0_tbl.hpp"
#undef NGRAPH_OP
#define NGRAPH_OP(a, b) {b::a::type_info, OP_TYPEID::a##_v1},
#include "ngraph/op/op_v1_tbl.hpp"
#undef NGRAPH_OP
#ifdef INTERPRETER_USE_HYBRID
#define NGRAPH_OP(a, b) {b::a::type_info, OP_TYPEID::a##_hybrid},
#include "ngraph/runtime/hybrid/op/op_tbl.hpp"
#undef NGRAPH_OP
#endif
    };
#undef NGRAPH_OP

    auto it = type_info_map.find(m_node->get_type_info());
    if (it != type_info_map.end())
    {
        m_typeid = it->second;
    }
    else
    {
        throw unsupported_op("Unsupported op '" + m_node->description() + "'");
    }
}
