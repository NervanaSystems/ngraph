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
                   const op::ParameterVector& parameters,
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

Function::Function(const NodeVector& results,
                   const op::ParameterVector& parameters,
                   const std::string& name)
    : m_results(results.size())
    , m_parameters(parameters)
    , m_temporary_pool_size(0)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_instance_id))
{
    if (std::any_of(results.cbegin(), results.cend(), [](std::shared_ptr<Node> n) {
            return std::dynamic_pointer_cast<op::Result>(n);
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
                   const op::ParameterVector& parameters,
                   const std::string& name)
    : Function(NodeVector{result}, parameters, name)
{
}

void Function::init()
{
    traverse_nodes(this, [&](shared_ptr<Node> node) {
        std::shared_ptr<op::Parameter> p = std::dynamic_pointer_cast<op::Parameter>(node);
        if (nullptr != p)
        {
            auto it = std::find_if(m_parameters.begin(),
                                   m_parameters.end(),
                                   [p](std::shared_ptr<op::Parameter> q) { return (p == q); });
            if (it == m_parameters.end())
            {
                throw ngraph_error("Function references undeclared parameter");
            }
        }
    });
}

std::list<shared_ptr<Node>> Function::get_ordered_ops()
{
    return topological_sort(get_ops());
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

void Function::set_name(const string& name)
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

std::ostream& ngraph::operator<<(std::ostream& out, const Function& f)
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

shared_ptr<Node> Function::get_output_op(size_t i) const
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

std::list<shared_ptr<Node>> Function::get_ops() const
{
    std::list<std::shared_ptr<Node>> ops;
    traverse_nodes(this, [&](shared_ptr<Node> node) { ops.push_back(node); });
    return ops;
}

void Function::replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl)
{
    ngraph::replace_node(old, repl);
}
