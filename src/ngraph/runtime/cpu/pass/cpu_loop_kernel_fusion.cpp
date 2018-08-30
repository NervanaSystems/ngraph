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

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/runtime/cpu/pass/cpu_loop_kernel_fusion.hpp"

#define TI(x) std::type_index(typeid(x))

using namespace ngraph;

struct LKGraph
{
    LKGraph(const NodeVector& ns, const NodeVector& ins)
        : m_inputs(ins)
        , m_nodes(ns)
    {
    }
    NodeVector m_inputs;
    NodeVector m_nodes;
};

class LoopKernelCollector
{
public:
    LoopKernelCollector(std::shared_ptr<Function> f, size_t min_nodes_to_fuse)
    {
        for (auto n : f->get_ordered_ops())
        {
            if (is_fusible(n))
            {
                auto arg_from_fusible_group = collect_fusible_args(n);
                // create a new group
                if (!arg_from_fusible_group)
                {
                    m_heads.insert(std::make_pair(n, n));
                    m_graphs.insert(std::make_pair(n, LKGraph{{n}, n->get_arguments()}));
                    NGRAPH_DEBUG << "Created a new group for " << n->get_name();
                    log_group(n);
                }
                else
                {
                    auto smallest_head = m_heads.at(arg_from_fusible_group);
                    auto& lkgraph = m_graphs.at(smallest_head);
                    lkgraph.m_nodes.push_back(n);
                    for (auto arg : n->get_arguments())
                    {
                        if (is_leaf(arg))
                        {
                            lkgraph.m_inputs.push_back(arg);
                        }
                    }
                    m_heads.insert(std::make_pair(n, smallest_head));
                    log_group(smallest_head);
                }
            }
        }

        prune_graphs(min_nodes_to_fuse);
    }

    const std::vector<std::shared_ptr<runtime::cpu::op::LoopKernel>> get_loop_kernels() const
    {
        std::vector<std::shared_ptr<runtime::cpu::op::LoopKernel>> lks;
        for (auto e : m_graphs)
        {
            auto& lkg = e.second;

            NodeVector member_outputs = ngraph::get_subgraph_outputs(lkg.m_nodes, NodeVector{});
            auto lk = std::make_shared<runtime::cpu::op::LoopKernel>(
                lkg.m_nodes, member_outputs, lkg.m_inputs);
            lks.push_back(lk);
        }
        return lks;
    }

private:
    static bool is_fusible(std::shared_ptr<Node> n)
    {
        static const std::set<std::type_index> fusible_ops_set{TI(ngraph::op::Abs),
                                                               TI(ngraph::op::Add),
                                                               TI(ngraph::op::Negative),
                                                               TI(ngraph::op::Subtract),
                                                               TI(ngraph::op::Relu),
                                                               TI(ngraph::op::Minimum),
                                                               TI(ngraph::op::Maximum)};

        const Node& node = *n;
        return fusible_ops_set.count(TI(node)) != 0;

        // return (std::dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(n) ||
        //         std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(n));
    }

    bool is_leaf(std::shared_ptr<Node> src) { return src->is_parameter() || src->is_constant(); }
    void prune_graphs(size_t min_nodes_to_fuse)
    {
        for (auto it = m_graphs.begin(); it != m_graphs.end();)
        {
            if (it->second.m_nodes.size() < min_nodes_to_fuse)
            {
                it = m_graphs.erase(it);
            }
            else
            {
                it++;
            }
        }
    }

    void log_group(std::shared_ptr<Node> head) const
    {
        NGRAPH_DEBUG << "Group leader : " << head->get_name() << std::endl;
        NGRAPH_DEBUG << "Group members : " << m_graphs.at(head).m_nodes << std::endl;
        NGRAPH_DEBUG << "Inputs: " << m_graphs.at(head).m_inputs << std::endl;
    }

    std::shared_ptr<Node> collect_fusible_args(std::shared_ptr<Node> n)
    {
        std::shared_ptr<Node> arg_from_fusible_group;
        for (auto arg : n->get_arguments())
        {
            // an argument is fusible and a part of some group
            NGRAPH_DEBUG << "Considering " << arg->get_name();
            if (m_heads.count(arg) != 0)
            {
                if (!arg_from_fusible_group)
                {
                    arg_from_fusible_group = arg;
                }
                else
                {
                    if (!is_leaf(arg) && m_heads.at(arg) != m_heads.at(arg_from_fusible_group))
                    {
                        return {nullptr};
                    }
                }
            }
        }
        return arg_from_fusible_group;
    }

    std::unordered_map<std::shared_ptr<Node>, LKGraph> m_graphs;
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> m_heads;
};

bool ngraph::runtime::cpu::pass::CPULoopKernelFusion::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    LoopKernelCollector lkc(function, m_min_kernel_size);
    auto loop_kernels = lkc.get_loop_kernels();

    for (auto lk : loop_kernels)
    {
        auto outputs = lk->get_kernel_outputs();
        std::set<std::shared_ptr<Node>> lk_nodes_set(lk->get_node_list().begin(),
                                                     lk->get_node_list().end());
        for (size_t i = 0; i < outputs.size(); i++)
        {
            auto ith_goe = std::make_shared<ngraph::op::GetOutputElement>(lk, i);
            auto& ith_output = ith_goe->get_outputs().at(0);

            if (outputs.at(i)->get_outputs().size() > 1)
            {
                throw ngraph_error(
                    "support for fusing multi-output nodes in loop kernels isn't yet implemented");
            }

            // TODO: revisit when we need support for multi-output nodes
            auto& orig_output = outputs.at(i)->get_outputs().at(0);

            // this is needed since replace_output modifies orig_output.get_inputs()
            std::set<ngraph::descriptor::Input*> inputs_copy{begin(orig_output.get_inputs()),
                                                             end(orig_output.get_inputs())};
            for (auto input : inputs_copy)
            {
                // this user is NOT internal to this loop kernel
                // so it needs to be replaced with corresponding lk's GOE
                if (lk_nodes_set.count(input->get_node()) == 0)
                {
                    input->replace_output(ith_output);
                }
            }
        }
    }

    return !lkc.get_loop_kernels().empty();
}
