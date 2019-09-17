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

#include <algorithm>
#include <list>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

atomic<size_t> Function::m_next_instance_id(0);

Function::Function(const ResultVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(results)
    , m_parameters(parameters)
    , m_temporary_pool_size(0)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_instance_id))
{
    init();
}

Function::Function(const OutputVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(results.size())
    , m_parameters(parameters)
    , m_temporary_pool_size(0)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_instance_id))
{
    if (std::any_of(results.cbegin(), results.cend(), [](Output<Node> n) {
            return as_type_ptr<op::Result>(n.get_node_shared_ptr());
        }))
    {
        throw ngraph_error(
            " Results already contain op::Results. Use a c-tor that takes a ResultVector");
    }

    std::transform(results.begin(), results.end(), m_results.begin(), [](Output<Node> n) {
        return std::make_shared<op::Result>(n);
    });
    init();
}

Function::Function(const NodeVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(results.size())
    , m_parameters(parameters)
    , m_temporary_pool_size(0)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_instance_id))
{
    if (std::any_of(results.cbegin(), results.cend(), [](std::shared_ptr<Node> n) {
            return as_type_ptr<op::Result>(n);
        }))
    {
        throw ngraph_error(
            " Results already contain op::Results. Use a c-tor that takes a ResultVector");
    }

    std::transform(results.begin(), results.end(), m_results.begin(), [](std::shared_ptr<Node> n) {
        return std::make_shared<op::Result>(n);
    });
    init();
}

Function::Function(const std::shared_ptr<Node>& result,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Function(NodeVector{result}, parameters, name)
{
}

void Function::validate_nodes_and_infer_types()
{
    ngraph::validate_nodes_and_infer_types(get_ops());
}

void Function::init()
{
    validate_nodes_and_infer_types();

    traverse_nodes(this,
                   [&](shared_ptr<Node> node) {
                       if (node->is_parameter())
                       {
                           auto it = std::find(m_parameters.begin(), m_parameters.end(), node);
                           if (it == m_parameters.end())
                           {
                               throw ngraph_error("Function references undeclared parameter");
                           }
                       }
                   },
                   true /*include control dependencies*/);
}

std::list<shared_ptr<Node>> Function::get_ordered_ops(bool include_control_deps) const
{
    NodeVector nodes;
    for (auto& r : get_results())
    {
        nodes.push_back(r);
    }
    for (auto& param : get_parameters())
    {
        nodes.push_back(param);
    }

    return topological_sort(nodes, include_control_deps);
}

void Function::map_unordered_ops(std::function<void(Node*)> f) const
{
    std::unordered_set<Node*> unordered_ops;
    std::stack<Node*, std::vector<Node*>> remaining_ops;
    for (auto& r : get_results())
    {
        remaining_ops.push(r.get());
    }
    for (auto& param : get_parameters())
    {
        remaining_ops.push(param.get());
    }
    while (remaining_ops.size() > 0)
    {
        Node* op = remaining_ops.top();
        remaining_ops.pop();
        if (unordered_ops.insert(op).second)
        {
            f(op);
            for (size_t i = 0; i < op->get_input_size(); ++i)
            {
                remaining_ops.push(op->input(i).get_source_output().get_node());
            }
            for (auto& cdep : op->get_control_dependencies())
            {
                remaining_ops.push(cdep.get());
            }
        }
    }
}

const std::string& Function::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Function::get_name() const
{
    return m_unique_name;
}

void Function::set_friendly_name(const string& name)
{
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph_error("Function name may be set exactly once");
    }
}

size_t Function::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void Function::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}

std::ostream& operator<<(std::ostream& out, const Function& f)
{
    out << "Function(" << f.get_name() << ")";
    return out;
}

size_t Function::get_output_size() const
{
    return m_results.size();
}

const element::Type& Function::get_output_element_type(size_t i) const
{
    return m_results.at(i)->get_element_type();
}

const Shape& Function::get_output_shape(size_t i) const
{
    return m_results.at(i)->get_shape();
}

const PartialShape& Function::get_output_partial_shape(size_t i) const
{
    return m_results.at(i)->get_output_partial_shape(0);
}

shared_ptr<Node> Function::get_output_op(size_t i) const
{
    return m_results.at(i);
}

Output<Node> Function::output(size_t i) const
{
    return m_results.at(i);
}

shared_ptr<Node> Function::get_result() const
{
    if (m_results.size() != 1)
    {
        throw ngraph_error("get_result() must be called on a function with exactly one result.");
    }
    return m_results.at(0);
}

std::list<shared_ptr<Node>> Function::get_ops(bool include_control_deps) const
{
    std::list<std::shared_ptr<Node>> ops;
    traverse_nodes(this, [&](shared_ptr<Node> node) { ops.push_back(node); }, include_control_deps);
    return ops;
}

void Function::replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl)
{
    ngraph::replace_node(old, repl);
}

size_t Function::get_graph_size() const
{
    size_t total_size = 0;
    for (auto node : get_ops())
    {
        total_size += sizeof(*node);
        if (node->description() == "Constant")
        {
            const Shape& shape = node->output(0).get_shape();
            size_t const_size = node->output(0).get_element_type().size();
            if (shape.size() == 0)
            {
                total_size += const_size;
            }
            else
            {
                total_size += (const_size * shape_size(node->output(0).get_shape()));
            }
        }
    }
    return total_size;
}

size_t Function::get_placement() const
{
    return m_placement;
}

void Function::set_placement(size_t placement)
{
    m_placement = placement;
}

// TODO(pthoreho) this will be expensive, since we will be traversing all the nodes in
// the graph, figure out if their is a way to cache the result and invalidate/update
// the result if the function is modified
bool Function::is_dynamic() const
{
    auto list_of_nodes = this->get_ops();
    for (auto& node : list_of_nodes)
    {
        if (node->get_output_partial_shape(0).is_dynamic())
        {
            return true;
        }
    }
    return false;
}

void Function::replace_parameter(size_t parameter_index, const shared_ptr<op::Parameter>& parameter)
{
    NGRAPH_CHECK(parameter_index < m_parameters.size(),
                 "replace_parameter(): Tried to replace parameter at index ",
                 parameter_index,
                 " but the function only has ",
                 m_parameters.size(),
                 " parameters.");
    replace_node(m_parameters[parameter_index], parameter);
    m_parameters[parameter_index] = parameter;
}
