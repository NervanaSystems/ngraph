//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/node_input.hpp"
#include "ngraph/node_output.hpp"

using namespace ngraph;

NodeOutput NodeInput::get_source_output() const
{
    auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
    return NodeOutput(output_descriptor.get_node(), output_descriptor.get_index());
}

void NodeInput::replace_source_output(const NodeOutput& new_source_output) const
{
    m_node->replace_input_source_output(
        m_index, new_source_output.get_node(), new_source_output.get_index());
}

void NodeInput::replace_source_output(const std::shared_ptr<Node>& new_source_node,
                                      size_t output_index) const
{
    m_node->replace_input_source_output(m_index, new_source_node, output_index);
}
