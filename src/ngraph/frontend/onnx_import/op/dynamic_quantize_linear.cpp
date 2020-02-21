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

#include <fstream>
#include <memory>

#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/runtime/backend.hpp"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "round.hpp"

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
                    // Create TypeProto
                    onnx::TypeProto float_type_shape_6;
                    float_type_shape_6.mutable_tensor_type()->set_elem_type(
                        onnx::TensorProto_DataType_FLOAT);
                    float_type_shape_6.mutable_tensor_type()
                        ->mutable_shape()
                        ->add_dim()
                        ->set_dim_value(6);

                    onnx::TypeProto uint8_type_shape_6;
                    uint8_type_shape_6.mutable_tensor_type()->set_elem_type(
                        onnx::TensorProto_DataType_UINT8);
                    uint8_type_shape_6.mutable_tensor_type()
                        ->mutable_shape()
                        ->add_dim()
                        ->set_dim_value(6);

                    onnx::TypeProto float_type_no_scalar;
                    float_type_no_scalar.mutable_tensor_type()->set_elem_type(
                        onnx::TensorProto_DataType_FLOAT);

                    onnx::TypeProto uint8_type_no_scalar;
                    uint8_type_no_scalar.mutable_tensor_type()->set_elem_type(
                        onnx::TensorProto_DataType_UINT8);

                    const onnx::NodeProto node_proto = node.node_proto();

                    // Create a graph
                    onnx::GraphProto graph;
                    onnx::NodeProto* new_node = graph.add_node();
                    new_node->CopyFrom(node_proto);
                    new_node->clear_input();
                    new_node->clear_output();

                    // Add inputs to node and graph
                    for (std::shared_ptr<ngraph::Node> input : node.get_ng_inputs())
                    {
                        new_node->add_input(input->get_name());
                        onnx::ValueInfoProto* proto_input = graph.add_input();
                        proto_input->set_name(input->get_name());
                        *proto_input->mutable_type() = float_type_shape_6;
                    }

                    // Add outputs to node
                    for (auto output : node.get_output_names())
                    {
                        new_node->add_output(output);
                    }

                    // Add outputs to graph
                    onnx::ValueInfoProto* y = graph.add_output();
                    y->set_name("y");
                    *y->mutable_type() = uint8_type_shape_6;

                    onnx::ValueInfoProto* y_scale = graph.add_output();
                    y_scale->set_name("y_scale");
                    *y_scale->mutable_type() = float_type_no_scalar;

                    onnx::ValueInfoProto* y_zero_point = graph.add_output();
                    y_zero_point->set_name("y_zero_point");
                    *y_zero_point->mutable_type() = uint8_type_no_scalar;

                    const auto* schema = onnx::OpSchemaRegistry::Schema(node.op_type(), 11, "");
                    const onnx::FunctionProto* func = schema->GetFunction();

                    FunctionExpandHelper(*new_node, *func, graph);

                    graph.mutable_node()->erase(graph.node().begin());

                    // Save graph to file
                    onnx::ModelProto model;
                    auto* graph_ptr = model.mutable_graph();
                    *graph_ptr = graph;
                    model.set_ir_version(5);
                    model.set_producer_name("backend-test");
                    auto* opset_version = model.add_opset_import();
                    opset_version->set_version(11);
                    const std::string path = "/home/etusien/ngraph/test/models/onnx/dql_test.onnx";
                    std::ofstream output_file{path};
                    model.SerializeToOstream(&output_file);

                    auto function = ngraph::onnx_import::import_onnx_model(path);
                    std::vector<std::shared_ptr<ngraph::Node>> nodes = function->get_ordered_ops();

                    // Delete parameters and outputs
                    for (int i = nodes.size() - 1; i >= 0; --i)
                    {
                        std::cout << nodes.at(i)->get_name() << std::endl;
                        if (nodes.at(i)->is_output() || nodes.at(i)->is_parameter())
                        {
                            nodes.erase(nodes.begin() + i);
                        }
                    }
                    return NodeVector{nodes};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph