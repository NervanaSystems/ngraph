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
#include "ngraph/pass/opset1_transform.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/softmax.hpp"

using namespace std;
using namespace ngraph;

// @TODO: Shouldn't this be moved to a common utility class? This mapping to OP_TYPEID gets repeated many times.
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

#define NGRAPH_OP(a, b) {#a, OP_TYPEID::a},
static unordered_map<string, OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

static OP_TYPEID get_typeid(shared_ptr<Node> node)
{
    OP_TYPEID type_id;
    auto it = typeid_map.find(node->description());
    if (it != typeid_map.end())
    {
        type_id = it->second;
    }
    else
    {
        throw unsupported_op("Unsupported op '" + node->description() + "'");
    }
    return type_id;
}
// END mapping to OP_TYPEID

bool pass::Opset1Transformation::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;

    size_t opset_version = node->get_opset_version();

    if (opset_version == 1)
    {
        return modified;
    }

    if (opset_version != 0)
    {
        throw ngraph_error("Opset 1 transformation pass failed for " + node->get_name() +
                           ", only opset 0 operations expected. Opset " + to_string(opset_version) +
                           " found.");
    }

    switch (get_typeid(node))
    {
    case OP_TYPEID::Softmax:
    {
        auto tmp = dynamic_cast<const op::set0::Softmax*>(node.get());
        AxisSet axes = tmp->get_axes();

        if (axes.size() != 1)
        {
            throw ngraph_error(
                "Unable to convert Softmax:0 to Softmax:1 with more then one axis. " +
                node->get_name() + ".");
        }

        auto replacement_node =
            make_shared<op::set1::Softmax>(node->input(0).get_source_output(), axes.to_vector()[0]);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    default:
    {
        node->set_opset_version(1);
        modified = true;
        break;
    }
    }

    return modified;
}
