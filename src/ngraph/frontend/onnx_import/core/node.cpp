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

#include "node.hpp"
#include "graph.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        NodeVector Node::get_ng_nodes() const { return m_graph->make_ng_nodes(*this); }
        NodeVector Node::get_ng_inputs() const
        {
            NodeVector result;
            for (const auto& name : m_node_proto->input())
            {
                result.push_back(m_graph->get_ng_node_from_cache(name));
            }
            return result;
        }

        std::string Node::get_description() const
        {
            if (!get_name().empty())
            {
                return get_name();
            }

            std::stringstream stream;
            for (std::size_t index = 0; index < m_output_names.size(); ++index)
            {
                stream << (index != 0 ? ", " : "");
                stream << m_output_names.at(index).get();
            }
            return stream.str();
        }

    } // namespace onnx_import

} // namespace ngraph
