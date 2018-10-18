//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "attribute.hpp"
#include "graph.hpp"
#include "operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::vector<Graph> Attribute::get_graph_array(const OperatorSet& opset) const
        {
            std::vector<Graph> result;
            for (const auto& graph : m_attribute_proto->graphs())
            {
                result.emplace_back(graph, opset);
            }
            return result;
        }

        Graph Attribute::get_graph(const OperatorSet& opset) const
        {
            return Graph{m_attribute_proto->g(), opset};
        }

    } // namespace onnx_import

} // namespace ngraph
