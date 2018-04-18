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

#include <ostream>
#include <vector>
#include "node.hpp"
#include "onnx.pb.h"
#include "value_info.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Graph
        {
            onnx::GraphProto m_graph_proto;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_values;

            friend std::ostream& operator<<(std::ostream& os, const Graph& wrapper);

        public:
            explicit Graph(const onnx::GraphProto& proto);
            const std::vector<Node>& get_nodes() const;
            const std::vector<ValueInfo>& get_values() const;
        };

        std::ostream& operator<<(std::ostream& os, const Graph& wrapper);

    } // namespace onnx_import
} // namespace ngraph
