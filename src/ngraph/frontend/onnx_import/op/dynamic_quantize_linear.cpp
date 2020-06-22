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

#include "dynamic_quantize_linear.hpp"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "utils/common.hpp"

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
                    const ONNX_NAMESPACE::NodeProto node_proto = node.node_proto();

                    // Create a graph
                    ONNX_NAMESPACE::GraphProto graph;
                    ONNX_NAMESPACE::NodeProto* new_node = graph.add_node();
                    new_node->CopyFrom(node_proto);
                    new_node->clear_input();
                    new_node->clear_output();

                    std::vector<std::shared_ptr<ngraph::Node>> orginal_inputs =
                        node.get_ng_inputs();
                    std::vector<std::shared_ptr<ngraph::Node>> helper_inputs;

                    // Add input to node and graph
                    for (auto input : orginal_inputs)
                    {
                        new_node->add_input(input->get_name());
                        ONNX_NAMESPACE::ValueInfoProto* proto_input = graph.add_input();
                        proto_input->set_name(input->get_name());
                        auto input_type = input->get_element_type();
                        auto input_shape = input->get_output_shape(0);
                        *proto_input->mutable_type() =
                            common::get_proto_type(input_type, input_shape);
                    }
                    // Warning: Consider PartialShape

                    // Add outputs to node
                    for (auto output : node.get_output_names())
                    {
                        new_node->add_output(output);

                        // Add outputs to graph
                        ONNX_NAMESPACE::ValueInfoProto* y = graph.add_output();
                        y->set_name(output);
                    }

                    // Swap input from helping nGrpah function with one from original function
                    std::vector<std::shared_ptr<ngraph::Node>> nodes =
                        common::get_extanded_function(new_node, graph, 11);

                    for (int i = nodes.size() - 1; i >= 0; --i)
                    {
                        if (nodes.at(i)->is_parameter())
                        {
                            helper_inputs.push_back(nodes.at(i));
                        }
                    }

                    std::vector<std::string> output_op_names;

                    for (int i = nodes.size() - 1; i >= 0; --i)
                    {
                        if (nodes.at(i)->is_output())
                        {
                            output_op_names.push_back(
                                nodes.at(i)->get_input_node_ptr(0)->get_name());
                        }
                        for (auto& input : nodes.at(i)->inputs())
                        {
                            for (int i = 0; i < helper_inputs.size(); ++i) // Func neeeded
                            {
                                if (input.get_source_output() == helper_inputs.at(i))
                                {
                                    input.replace_source_output(orginal_inputs.at(i));
                                }
                            }
                        }
                    }

                    NodeVector final_nodes;
                    for (int i = nodes.size() - 1; i >= 0; --i)
                    {
                        for (auto name : output_op_names)
                        {
                            if (name == nodes.at(i)->get_name())
                            {
                                final_nodes.push_back(nodes.at(i));
                            }
                        }
                    }
                    return final_nodes;
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
