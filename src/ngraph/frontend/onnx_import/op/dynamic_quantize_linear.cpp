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

#include <memory>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "round.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector dynamic_quantize_linear(const Node& node)
                {
                    onnx::GraphProto graph;
                    onnx::NodeProto* new_node = graph.add_node();
                    new_node->set_op_type(node.op_type());
                    for(auto node_input : node.get_ng_inputs())
                    {
                        new_node->add_input(node_input->get_name());
                    }

                    for(auto node_output : node.get_output_names())
                    {
                        new_node->add_output(node_output);
                    }
                    const auto* schema = onnx::OpSchemaRegistry::Schema(node.op_type(), 9, "");
                    const onnx::FunctionProto* func = schema->GetFunction();

                    FunctionExpandHelper(*new_node, *func, graph);

                    onnx::ModelProto model;
                    auto* graph_ptr = model.mutable_graph();
                    *graph_ptr = graph;
                    auto function = ngraph::onnx_import::import_onnx_proto_model(model);

                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph