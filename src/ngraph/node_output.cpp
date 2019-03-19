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

#include "ngraph/node_output.hpp"
#include "ngraph/node_input.hpp"

using namespace ngraph;

std::set<NodeInput> NodeOutput::get_target_inputs() const
{
    return m_node->get_output_target_inputs(m_index);
}

void NodeOutput::remove_target_input(const NodeInput& target_input) const
{
    m_node->remove_output_target_input(m_index, target_input);
}
