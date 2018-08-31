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

#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node>
    ngraph::runtime::cpu::op::LoopKernel::copy_with_new_args(const NodeVector& new_args) const
{
    auto args = get_arguments();
    if (new_args.size() != args.size())
    {
        throw ngraph_error("number of arguments don't match");
    }

    // map inputs
    NodeMap nm;
    for (size_t i = 0; i < args.size(); i++)
    {
        nm.add(args.at(i), new_args.at(i));
    }

    NodeVector new_node_list;
    for (auto n : m_node_list)
    {
        NodeVector cur_args;
        for (auto a : n->get_arguments())
        {
            cur_args.push_back(nm.get(a));
        }
        auto new_n = n->copy_with_new_args(cur_args);
        nm.add(n, new_n);
        new_node_list.push_back(new_n);
    }

    NodeVector new_outputs;
    for (auto o : m_output_nodes)
    {
        new_outputs.push_back(nm.get(o));
    }

    return std::make_shared<LoopKernel>(new_node_list, new_outputs, new_args);
}

ngraph::runtime::cpu::op::LoopKernel::LoopKernel(const NodeVector& node_list,
                                                 const NodeVector& outputs,
                                                 const NodeVector& args)
    : Op("LoopKernel", check_single_output_args({args}))
    , m_node_list(node_list)
    , m_output_nodes(outputs)
{
    constructor_validate_and_infer_types();
    set_output_size(m_output_nodes.size());

    auto ref = node_list.at(0);
    for (auto n : node_list)
    {
        if (n->get_shape() != ref->get_shape() || n->get_element_type() != ref->get_element_type())
        {
            throw ngraph_error("types and shapes of the nodes in node_list are different");
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto& o = outputs.at(i);

        if (std::find(node_list.begin(), node_list.end(), o) == node_list.end())
        {
            throw ngraph_error(o->get_name() + " isn't in node_list");
        }
        set_output_type(i, o->get_element_type(), o->get_shape());
    }
}
