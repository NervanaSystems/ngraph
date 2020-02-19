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
#include <fstream>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "round.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/runtime/backend.hpp"


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
                    const auto& model_proto=  node.model();
                    const onnx::NodeProto node_proto = node.node_proto();
                    onnx::NodeProto* new_node = graph.add_node();
                    *new_node = node_proto;
                    std::cout<<"schema"<<std::endl;
                    const auto* schema = onnx::OpSchemaRegistry::Schema(node.op_type(), 11, "");
                    const onnx::FunctionProto* func = schema->GetFunction();

                    FunctionExpandHelper(node_proto, *func, graph);

                     graph.mutable_node()->erase(graph.node().begin());

                    std::cout<<"model"<<std::endl;
                    onnx::ModelProto model = *model_proto; 
                    auto* graph_ptr = model.mutable_graph();
                    *graph_ptr = graph;
                    const std::string path = "/home/etusien/ngraph/test/models/onnx/dql_test.onnx";
                    std::ofstream output_file{path};
                    model.SerializeToOstream(&output_file);
                    std::cout<<"save"<<std::endl;
                    auto function = ngraph::onnx_import::import_onnx_model(path);
                    std::cout<<"function"<<std::endl;
                    std::vector<std::shared_ptr<ngraph::Node>> nodes = function->get_ordered_ops();

                   return NodeVector{nodes};

                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph