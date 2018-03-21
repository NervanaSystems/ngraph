/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;

std::shared_ptr<Node> make_zero(const element::Type& element_type, const Shape& shape)
{
    std::shared_ptr<Node> zero = op::Constant::create(element_type, Shape{}, {0.0});
    if (shape.size() > 0)
    {
        AxisSet axes;
        for (size_t i = 0; i < shape.size(); i++)
        {
            axes.insert(i);
        }
        zero = std::make_shared<op::Broadcast>(zero, shape, axes);
    }
    return zero;
}

autodiff::Adjoints::Adjoints(const std::shared_ptr<Node>& y, const std::shared_ptr<Node>& c)
{
    // Pass 1 determines which nodes contribute to y as well as setting up a reverse
    // topological sort.

    // Number of nodes that use the node's value
    std::unordered_map<std::shared_ptr<Node>, size_t> parent_counts;

    // Nodes that have been processed
    std::unordered_set<std::shared_ptr<Node>> visited_nodes;

    // Nodes we should check
    std::list<std::shared_ptr<Node>> nodes_to_check;
    nodes_to_check.push_front(y);
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        if (visited_nodes.count(node) != 0)
        {
            continue;
        }
        for (auto arg : node->get_input_ops())
        {
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
    m_adjoint_map[y.get()] = c;
    nodes_to_check.push_front(y);
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        // Look for nodes that will be available when this node is done
        for (auto arg : node->get_input_ops())
        {
            auto count_it = parent_counts.find(arg);
            count_it->second--;
            if (0 == count_it->second)
            {
                nodes_to_check.push_front(arg);
            }
        }
        node->generate_adjoints(*this, get(node));
    }
}

std::shared_ptr<Node> autodiff::Adjoints::get(const std::shared_ptr<Node>& x)
{
    auto adjoint_it = m_adjoint_map.find(x.get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto& output = x->get_outputs().at(0);
        auto result = make_zero(output.get_element_type(), output.get_shape());
        adjoint_it = m_adjoint_map.insert({x.get(), result}).first;
    }
    return adjoint_it->second;
}

void autodiff::Adjoints::add_delta(const std::shared_ptr<Node>& x,
                                   const std::shared_ptr<Node>& delta)
{
    if (!x->has_same_type(delta) && delta->get_shape() != x->get_outputs().at(0).get_shape())
    {
        throw ngraph_error("Autodiff internal error: Mismatch on backprop and op in add_delta.");
    }
    auto adjoint_it = m_adjoint_map.find(x.get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        m_adjoint_map.insert({x.get(), delta});
    }
    else
    {
        adjoint_it->second = std::make_shared<op::Add>(adjoint_it->second, delta);
    }
}

void autodiff::Adjoints::add_delta_to_slice(const std::shared_ptr<Node>& x,
                                            const std::shared_ptr<Node>& delta,
                                            const Coordinate& lower_bounds,
                                            const Coordinate& upper_bounds,
                                            const Strides& strides)
{
    if (x->get_output_size() != 1 || delta->get_output_size() != 1 ||
        x->get_output_element_type(0) != delta->get_output_element_type(0) ||
        x->get_output_shape(0).size() != delta->get_output_shape(0).size())
    {
        throw ngraph_error(
            "Autodiff internal error: Mismatch on backprop and op in add_delta_to_slice.");
    }

    auto adjoint_it = m_adjoint_map.find(x.get());
    if (m_adjoint_map.end() == adjoint_it)
    {
        auto& output = x->get_outputs().at(0);
        auto zeros = make_zero(output.get_element_type(), output.get_shape());
        m_adjoint_map.insert({x.get(),
                              std::make_shared<op::ReplaceSlice>(
                                  zeros, delta, lower_bounds, upper_bounds, strides)});
    }
    else
    {
        adjoint_it->second = std::make_shared<op::ReplaceSlice>(
            adjoint_it->second,
            std::make_shared<op::Slice>(adjoint_it->second, lower_bounds, upper_bounds, strides) +
                delta,
            lower_bounds,
            upper_bounds,
            strides);
    }
}
