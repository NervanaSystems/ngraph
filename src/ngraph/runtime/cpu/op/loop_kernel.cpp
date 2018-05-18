/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node>
    ngraph::runtime::cpu::op::LoopKernel::copy_with_new_args(const NodeVector& new_args) const
{
    //TODO: this needs to call clone_nodes
    //repopulate outputs correctly
    throw ngraph_error("NYI");
}

ngraph::runtime::cpu::op::LoopKernel::LoopKernel(const NodeVector& node_list,
                                                 const NodeVector& outputs,
                                                 const NodeVector& args)
    : RequiresTensorViewArgs("LoopKernel", {args})
    , m_node_list(node_list)
    , m_outputs(outputs)
{
    auto ref = node_list.at(0);
    for (auto n : node_list)
    {
        if (n->get_shape() != ref->get_shape() || n->get_element_type() != ref->get_element_type())
        {
            throw ngraph_error("types and shapes of the nodes in node_list are different");
        }
    }

    for (auto o : outputs)
    {
        if (std::find(node_list.begin(), node_list.end(), o) == node_list.end())
        {
            throw ngraph_error(o->get_name() + " isn't in node_list");
        }
        add_output(o->get_element_type(), o->get_shape());
    }
}

/*
shared_ptr<Node> size_t op::LoopKernel::get_output_index(std::shared_ptr<Node> output)
{
    auto it = std::find(m_outputs.begin(), m_outputs.end(), output);
    if (it == std::end())
    {
        throw ngraph_error("node isn't in outputs");
    }
    
    return std::distance(m_outputs.begin(), it);
}
*/
