/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/add.hpp"
#include "node.hpp"
#include "graph.hpp"
#include "ops_bridge.hpp"

using namespace ngraph;

static NodeVector Add(ngraph::onnx_import::Node* onnx_node, ngraph::NodeVector ng_inputs)
{
    ngraph::NodeVector output_nodes{};
    output_nodes.push_back(ng_inputs[0] + ng_inputs[1]);
    return output_nodes;
}

NodeVector ngraph::onnx_import::ops_bridge::make_ng_nodes(ngraph::onnx_import::Node* onnx_node)
{
    std::string op_type = onnx_node->get_proto().op_type();
    ngraph::NodeVector ng_inputs = onnx_node->get_ng_inputs();

    if(op_type == "Add") return Add(onnx_node, ng_inputs);

    throw ngraph::ngraph_error("Unknown operation: " + op_type);
}
