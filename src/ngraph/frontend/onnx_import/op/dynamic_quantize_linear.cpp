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
                ONNX_NAMESPACE::TypeProto get_proto_type(element::Type type, Shape shape)
                {
                    ONNX_NAMESPACE::TypeProto target_type;
                    switch (type)
                    {
                    case element::Type_t::f32:
                        target_type.mutable_tensor_type()->set_elem_type(
                            ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
                        break;
                    case element::Type_t::u8:
                        target_type.mutable_tensor_type()->set_elem_type(
                            ONNX_NAMESPACE::TensorProto_DataType_UINT8);
                        break;
                    }

                    for (auto dim : shape)
                    {
                        target_type.mutable_tensor_type()
                            ->mutable_shape()
                            ->add_dim()
                            ->set_dim_value(dim);
                    }
                    return target_type;
                }

                NodeVector dynamic_quantize_linear(const Node& node)
                {
                    const ONNX_NAMESPACE::NodeProto node_proto = node.node_proto();

                    // Create a graph
                    ONNX_NAMESPACE::GraphProto graph;
                    ONNX_NAMESPACE::NodeProto* new_node = graph.add_node();
                    new_node->CopyFrom(node_proto);
                    new_node->clear_input();
                    new_node->clear_output();

                    // Add input to node and graph
                    auto input = node.get_ng_inputs().at(0);
                    new_node->add_input(input->get_name());
                    ONNX_NAMESPACE::ValueInfoProto* proto_input = graph.add_input();
                    proto_input->set_name(input->get_name());
                    auto input_type = input->get_element_type();
                    auto input_shape = input->get_output_shape(0);
                    *proto_input->mutable_type() = get_proto_type(input_type, input_shape);
                    // Warning: Consider PartialShape

                    // Add outputs to node
                    for (auto output : node.get_output_names())
                    {
                        new_node->add_output(output);
                    }

                    // Add outputs to graph
                    // This part is secific for each func op
                    ONNX_NAMESPACE::ValueInfoProto* y = graph.add_output();
                    y->set_name(node.get_output_names()[0]);
                    *y->mutable_type() = get_proto_type(element::Type_t::u8, input_shape);

                    ONNX_NAMESPACE::ValueInfoProto* y_scale = graph.add_output();
                    y_scale->set_name(node.get_output_names()[1]);
                    *y_scale->mutable_type() = get_proto_type(input_type, Shape(1));

                    ONNX_NAMESPACE::ValueInfoProto* y_zero_point = graph.add_output();
                    y_zero_point->set_name(node.get_output_names()[2]);
                    *y_zero_point->mutable_type() = get_proto_type(element::Type_t::u8, Shape(1));

                    const auto* schema =
                        ONNX_NAMESPACE::OpSchemaRegistry::Schema(node.op_type(), 11, "");
                    const ONNX_NAMESPACE::FunctionProto* func = schema->GetFunction();

                    FunctionExpandHelper(*new_node, *func, graph);

                    graph.mutable_node()->erase(graph.node().begin());

                    // Save graph to file
                    ONNX_NAMESPACE::ModelProto model;
                    auto* graph_ptr = model.mutable_graph();
                    *graph_ptr = graph;
                    model.set_ir_version(5);
                    model.set_producer_name("backend-test");
                    auto* opset_version = model.add_opset_import();
                    opset_version->set_version(11);
                    std::istringstream model_stream{model.SerializeAsString()};

                    auto function = ngraph::onnx_import::import_onnx_model(model_stream);
                    std::vector<std::shared_ptr<ngraph::Node>> nodes = function->get_ordered_ops();

                    for (int i = nodes.size() - 1; i >= 0; --i)
                    {
                        for (auto& input : nodes.at(i)->inputs())
                        {
                            if (input.get_shape() == node.get_ng_inputs().at(0)->get_shape())
                            {
                                input.replace_source_output(node.get_ng_inputs().at(0));
                            }
                        }
                    }
                    // Quantze, Divde, Convert
                    return NodeVector{nodes.at(23), nodes.at(10), nodes.at(16)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
