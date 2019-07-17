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

#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/strides.hpp"

using namespace ngraph;

Output<Node> make_broadcast_zero(const Output<Node>& output)
{
    Output<Node> zero = std::make_shared<op::ScalarConstantLike>(output, 0.0);
    Output<Node> bzero = std::make_shared<op::BroadcastLike>(zero, output, AxisSet{});
    return bzero;
}

OutputVector make_zeros(std::shared_ptr<Node> x)
{
    OutputVector zeros;
    for (auto output : x->outputs())
    {
        zeros.push_back(make_broadcast_zero(output));
    }
    return zeros;
}

autodiff::Adjoints::Adjoints(const NodeVector& ys, const NodeVector& cs)
    : Adjoints(OutputVector(ys.begin(), ys.end()), OutputVector(cs.begin(), cs.end()))
{
}

autodiff::Adjoints::Adjoints(const OutputVector& ys, const OutputVector& cs)
{
    if (ys.size() != cs.size())
    {
        throw ngraph_error("ys and cs must be equal size");
    }

    // Pass 1 determines which nodes contribute to y as well as setting up a reverse
    // topological sort.

    // Number of nodes that use the node's value
    std::unordered_map<std::shared_ptr<Node>, size_t> parent_counts;

    // Nodes that have been processed
    std::unordered_set<std::shared_ptr<Node>> visited_nodes;

    // Nodes we should check
    std::list<std::shared_ptr<Node>> nodes_to_check;
    for (auto& y : ys)
    {
        nodes_to_check.push_back(y.get_node_shared_ptr());
    }
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        if (visited_nodes.count(node) != 0)
        {
            continue;
        }
        for (auto input : node->inputs())
        {
            auto arg = input.get_source_output().get_node_shared_ptr();
            auto count_it = parent_counts.find(arg);
            if (count_it == parent_counts.end())
            {
                parent_counts[arg] = 1;
                nodes_to_check.push_front(arg);
            }
            else
            {
                parent_counts[arg]++;
            }
        }
        visited_nodes.insert(node);
    }

    // Second pass visits the nodes so that all users of a node's value are visited
    // before a node is visited.
    for (size_t i = 0; i < ys.size(); i++)
    {
        Node* n = ys.at(i).get_node();
        OutputVector t{cs.at(i)};
        std::pair<Node*, OutputVector> pair = std::make_pair(n, t);
        m_adjoint_map.insert(std::make_pair(ys.at(i).get_node(), OutputVector{cs.at(i)}));
    }

    for (auto& y : ys)
    {
        nodes_to_check.push_back(y.get_node_shared_ptr());
    }

    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        // Look for nodes that will be available when this node is done
        for (auto arg : node->get_arguments())
        {
            auto count_it = parent_counts.find(arg);
            count_it->second--;
            if (0 == count_it->second)
            {
                nodes_to_check.push_front(arg);
            }
        }
        OutputVector deltas = get(node);
        NodeVector delta_nodes;
        for (auto delta : deltas)
        {
            delta_nodes.push_back(get_output_element(delta));
        }
        node->generate_adjoints(*this, delta_nodes);
    }
}

const OutputVector& autodiff::Adjoints::get(const Output<Node>& x)
{
    auto adjoint_it = m_adjoint_map.find(x.get_node());
    if (m_adjoint_map.end() == adjoint_it)
    {
        adjoint_it =
            m_adjoint_map.insert({x.get_node(), make_zeros(x.get_node_shared_ptr())}).first;
    }
    return adjoint_it->second;
}

void autodiff::Adjoints::add_delta(const Output<Node>& x,
                                   const Output<Node>& delta,
                                   size_t output_index)
{
    auto adjoint_it = m_adjoint_map.find(x.get_node());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto zeros = make_zeros(x.get_node_shared_ptr());
        zeros.at(output_index) = delta;
        m_adjoint_map.insert({x.get_node(), zeros});
    }
    else
    {
        auto& deltas = adjoint_it->second;
        deltas.at(output_index) = std::make_shared<op::Add>(deltas.at(output_index), delta);
        adjoint_it->second = deltas;
    }
}

//This doesn't need an index since slice can only sit on top of GOE
void autodiff::Adjoints::add_delta_to_slice(const Output<Node>& x,
                                            const Output<Node>& delta,
                                            const Coordinate& lower_bounds,
                                            const Coordinate& upper_bounds,
                                            const Strides& strides)
{
    if (!(x.get_element_type().compatible(delta.get_element_type())) ||
        !(x.get_partial_shape().rank().compatible(delta.get_partial_shape().rank())))
    {
        throw ngraph_error(
            "Autodiff internal error: Mismatch on backprop and op in add_delta_to_slice.");
    }

    auto adjoint_it = m_adjoint_map.find(x.get_node());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto zero = make_broadcast_zero(x);
        OutputVector zeros{
            std::make_shared<op::ReplaceSlice>(zero, delta, lower_bounds, upper_bounds, strides)};
        m_adjoint_map.insert({x.get_node(), zeros});
    }
    else
    {
        auto& deltas = adjoint_it->second;
        deltas.at(0) = std::make_shared<op::ReplaceSlice>(
            deltas.at(0),
            std::make_shared<op::Add>(
                std::make_shared<op::Slice>(deltas.at(0), lower_bounds, upper_bounds, strides),
                delta),
            lower_bounds,
            upper_bounds,
            strides);
    }
}

std::shared_ptr<Node> autodiff::Adjoints::backprop_node(const Output<Node>& x)
{
    return get_output_element(backprop_output(x));
}

Output<Node> autodiff::Adjoints::backprop_output(const Output<Node>& x)
{
    return get(x).at(x.get_index());
}
