// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph;

size_t Function::m_next_instance_id = 0;

Function::Function(const std::shared_ptr<Node>& result,
                   const std::shared_ptr<ValueType>& result_type,
                   const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                   const std::string& name)
    : m_result(result)
    , m_parameters(parameters)
    , m_name(name)
    , m_result_type(result_type)
    , m_ordered_ops_valid(false)
    , m_instance_id(m_next_instance_id++)
{
    size_t i = 0;
    for (auto parameter : parameters)
    {
        parameter->assign_function(this, i++);
    }
}

void Function::set_ordered_ops(const std::list<Node*>& ordered_ops)
{
    m_ordered_ops = ordered_ops;
    m_ordered_ops_valid = true;
}

std::list<Node*>& Function::get_ops()
{
    return m_ops;
}

const std::list<Node*>& Function::get_ops() const
{
    return m_ops;
}

std::list<Node*>& Function::get_ordered_ops()
{
    if (!m_ordered_ops_valid)
    {
        throw ngraph_error("Access to ordered ops invalid");
    }
    return m_ordered_ops;
}

const std::list<Node*>& Function::get_ordered_ops() const
{
    if (!m_ordered_ops_valid)
    {
        throw ngraph_error("Access to ordered ops invalid");
    }
    return m_ordered_ops;
}

const std::string& Function::get_name() const
{
    if (m_name.empty())
    {
        (string) m_name = "Function_" + to_string(m_instance_id);
    }
    return m_name;
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

std::ostream& ngraph::operator<<(std::ostream& out, const Function& f)
{
    out << "Function(" << f.get_name() << ")";
    return out;
}
