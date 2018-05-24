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

#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>
#include "node.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"
#include "value_info.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Graph
        {
            onnx::GraphProto m_graph_proto;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_inputs;
            std::vector<ValueInfo> m_outputs;
            ngraph::op::ParameterVector m_parameters;
            std::map<std::string, std::shared_ptr<ngraph::Node>> m_ng_node_cache;

            friend std::ostream& operator<<(std::ostream& os, const Graph& wrapper);

        public:
            explicit Graph(const onnx::GraphProto& proto);
            const std::vector<Node>& get_nodes() const;
            const std::vector<ValueInfo>& get_inputs() const;
            const std::vector<ValueInfo>& get_outputs() const;
            const Tensor get_initializer(std::string) const;
            const ngraph::op::ParameterVector get_ng_parameters();
            const std::shared_ptr<ngraph::Node> get_ng_node_from_cache(std::string);
        };

        std::ostream& operator<<(std::ostream& os, const Graph& wrapper);

    } // namespace onnx_import
} // namespace ngraph
