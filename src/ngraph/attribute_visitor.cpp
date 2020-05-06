//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/attribute_adapter.hpp"

using namespace std;
using namespace ngraph;

void AttributeVisitor::on_attribute(const string& name, string& value)
{
    AttributeAdapter<string> handler(value);
    on_adapter(name, handler);
}

void AttributeVisitor::on_attribute(const string& name, bool& value)
{
    AttributeAdapter<bool> handler(value);
    on_adapter(name, handler);
}

void AttributeVisitor::start_structure(const string& name)
{
    m_context.push_back(Context{ContextType::Struct, name, 0});
}

void AttributeVisitor::finish_structure()
{
    m_context.pop_back();
}

void AttributeVisitor::start_vector(const std::string& name)
{
    m_context.push_back(Context{ContextType::Vector, name, 0});
}

void AttributeVisitor::next_vector_element()
{
    m_context.back().index++;
}

void AttributeVisitor::finish_vector()
{
    m_context.pop_back();
}

string AttributeVisitor::get_name_with_context(const std::string& name)
{
    ostringstream result;
    string sep = "";
    for (auto c : m_context)
    {
        result << sep;
        sep = ".";
        result << c.name;
        if (c.context_type == ContextType::Vector)
        {
            result << "[" << c.index << "]";
        }
    }
    result << sep << name;
    return result.str();
}

void AttributeVisitor::on_adapter(const std::string& name, VisitorAdapter& adapter)
{
    adapter.visit_attributes(*this, name);
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<string>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<bool>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<int8_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<int16_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<int32_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<int64_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint8_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint16_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint32_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint64_t>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<float>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<double>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int8_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int16_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int32_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int64_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint8_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint16_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint32_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint64_t>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<float>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<double>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<string>>& adapter)
{
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

const AttributeVisitor::node_id_t AttributeVisitor::invalid_node_id = "";

void AttributeVisitor::register_node(const std::shared_ptr<Node>& node, node_id_t id)
{
    m_id_node_map[id] = node;
    m_node_id_map[node] = id;
}

std::shared_ptr<Node> AttributeVisitor::get_registered_node(node_id_t id)
{
    auto it = m_id_node_map.find(id);
    return it == m_id_node_map.end() ? shared_ptr<Node>() : it->second;
}

AttributeVisitor::node_id_t
    AttributeVisitor::get_registered_node_id(const std::shared_ptr<Node>& node)
{
    auto it = m_node_id_map.find(node);
    return it == m_node_id_map.end() ? "" : it->second;
}
