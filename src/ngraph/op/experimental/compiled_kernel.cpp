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

#include "ngraph/op/experimental/compiled_kernel.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace std;
using namespace ngraph;

const string op::CompiledKernel::type_name{"CompiledKernel"};

shared_ptr<Node> ngraph::op::CompiledKernel::copy_with_new_args(const NodeVector& new_args) const
{
    auto args = input_values();
    if (new_args.size() != args.size())
    {
        throw ngraph_error("number of arguments don't match");
    }

    // map inputs
    NodeMap nm;
    for (size_t i = 0; i < args.size(); i++)
    {
        nm[args.at(i).get_node()] = new_args.at(i);
    }

    NodeVector new_node_list;
    for (auto n : m_node_list)
    {
        OutputVector cur_args;
        for (auto a : n->input_values())
        {
            if (dynamic_cast<ngraph::pattern::op::Label*>(a.get_node()))
            {
                // Label
                cur_args.push_back(a);
            }
            else
            {
                cur_args.push_back(a.for_node(nm.at(a.get_node())));
            }
        }
        auto new_n = n->copy_with_new_inputs(cur_args);
        nm[n.get()] = new_n;
        new_node_list.push_back(new_n);
    }

    NodeVector new_outputs;
    for (auto o : m_output_nodes)
    {
        new_outputs.push_back(nm.at(o.get()));
    }

    auto ck = std::make_shared<CompiledKernel>(new_node_list, new_outputs, new_args);
    for (auto tuple : m_node_arg_index_ck_arg_index)
    {
        ck->m_node_arg_index_ck_arg_index.push_back(
            std::tuple<std::shared_ptr<Node>, size_t, size_t>(
                nm[std::get<0>(tuple).get()], std::get<1>(tuple), std::get<2>(tuple)));
    }
    return ck;
}

ngraph::op::CompiledKernel::CompiledKernel(const OutputVector& node_list,
                                           const OutputVector& outputs,
                                           const OutputVector& args)
    : CompiledKernel(as_node_vector(node_list), as_node_vector(outputs), as_node_vector(args))
{
}

ngraph::op::CompiledKernel::CompiledKernel(const NodeVector& node_list,
                                           const NodeVector& outputs,
                                           const NodeVector& args)
    : Op(check_single_output_args({args}))
    , m_node_list(node_list)
    , m_output_nodes(outputs)
{
    constructor_validate_and_infer_types();
    set_output_size(m_output_nodes.size());

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

void ngraph::op::CompiledKernel::insert_to_vector(std::shared_ptr<Node> node,
                                                  size_t node_arg_idx,
                                                  size_t ck_arg_idx)
{
    m_node_arg_index_ck_arg_index.push_back(
        std::tuple<std::shared_ptr<Node>, size_t, size_t>(node, node_arg_idx, ck_arg_idx));
}
