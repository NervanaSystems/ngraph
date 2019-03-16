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

#include <cassert>
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

NodeOutput make_zero(const NodeOutput& node)
{
    std::shared_ptr<Node> zero = std::make_shared<op::ScalarConstantLike>(node, 0.0);
    std::shared_ptr<Node> bzero = std::make_shared<op::BroadcastLike>(zero, node, AxisSet{});
    return bzero;
}

OutputVector make_zeros(std::shared_ptr<Node> x)
{
    OutputVector zeros;
    for (size_t i = 0; i < x->get_output_size(); ++i)
    {
        zeros.push_back(make_zero(NodeOutput(x, i)));
    }
    return zeros;
}

autodiff::Adjoints::Adjoints(const OutputVector& ys, const OutputVector& cs)
{
    NGRAPH_ASSERT(ys.size() == cs.size());

    // Pass 1 determines which nodes contribute to y as well as setting up a reverse
    // topological sort.

    // Number of nodes that use the node's value
    std::unordered_map<std::shared_ptr<Node>, size_t> parent_counts;

    // Nodes that have been processed
    std::unordered_set<std::shared_ptr<Node>> visited_nodes;

    // Nodes we should check
    std::list<std::shared_ptr<Node>> nodes_to_check;
    for (auto& output : ys)
    {
        nodes_to_check.push_back(output.get_node());
    }
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        if (visited_nodes.count(node) != 0)
        {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); i++)
        {
            auto source_node = node->get_input_source_output(i).get_node();
            auto count_it = parent_counts.find(source_node);
            if (count_it == parent_counts.end())
            {
                parent_counts[source_node] = 1;
                nodes_to_check.push_front(source_node);
            }
            else
            {
                parent_counts[source_node]++;
            }
        }
        visited_nodes.insert(node);
    }

    // Second pass visits the nodes so that all users of a node's value are visited
    // before a node is visited.
    for (size_t i = 0; i < ys.size(); i++)
    {
        m_adjoint_map.insert(std::make_pair(ys.at(i).get_node().get(), OutputVector{cs.at(i)}));
    }

    nodes_to_check.clear();
    for (auto& output : ys)
    {
        nodes_to_check.push_back(output.get_node());
    }
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();

        // Look for nodes that will be available when this node is done
        for (size_t i = 0; i < node->get_input_size(); i++)
        {
            auto source_node = node->get_input_source_output(i).get_node();
            auto count_it = parent_counts.find(source_node);
            count_it->second--;
            if (0 == count_it->second)
            {
                nodes_to_check.push_front(source_node);
            }
        }

        node->build_backprop(*this, get(node));
    }
}

const OutputVector& autodiff::Adjoints::get(const std::shared_ptr<Node>& x)
{
    auto adjoint_it = m_adjoint_map.find(x.get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        adjoint_it = m_adjoint_map.insert({x.get(), make_zeros(x)}).first;
    }
    return adjoint_it->second;
}

void autodiff::Adjoints::add_output_delta(const NodeOutput& x, const NodeOutput& delta)
{
    auto adjoint_it = m_adjoint_map.find(x.get_node().get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto zeros = make_zeros(x.get_node());
        zeros.at(x.get_index()) = delta;
        m_adjoint_map.insert({x.get_node().get(), zeros});
    }
    else
    {
        auto& deltas = adjoint_it->second;
        deltas.at(x.get_index()) = std::make_shared<op::Add>(deltas.at(x.get_index()), delta);
        adjoint_it->second = deltas;
    }
}

void autodiff::Adjoints::add_output_delta_to_slice(const NodeOutput& x,
                                                   const NodeOutput& delta,
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

    auto adjoint_it = m_adjoint_map.find(x.get_node().get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto zeros = make_zeros(x.get_node());
        zeros.at(x.get_index()) = std::make_shared<op::ReplaceSlice>(
            zeros.at(x.get_index()), delta, lower_bounds, upper_bounds, strides);
        m_adjoint_map.insert({x.get_node().get(), zeros});
    }
    else
    {
        auto& deltas = adjoint_it->second;
        deltas.at(x.get_index()) = std::make_shared<op::ReplaceSlice>(
            deltas.at(x.get_index()),
            std::make_shared<op::Slice>(
                deltas.at(x.get_index()), lower_bounds, upper_bounds, strides) +
                delta,
            lower_bounds,
            upper_bounds,
            strides);
    }
}

std::shared_ptr<Node> autodiff::Adjoints::backprop_node(const std::shared_ptr<Node>& x)
{
    auto deltas = get(x);
    if (deltas.size() > 1)
    {
        throw ngraph_error("backprop_node is called for multi-output node");
    }
    return get_output_element(deltas.at(0).get_node(), deltas.at(0).get_index());
}
