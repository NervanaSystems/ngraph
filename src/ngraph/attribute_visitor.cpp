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

using namespace std;
using namespace ngraph;

void AttributeVisitor::start_structure(const string& name)
{
    m_context.push_back(Context{ContextType::Struct, name, "", 0});
}

void AttributeVisitor::finish_structure()
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
    }
    result << sep << name;
    return result.str();
}

void AttributeVisitor::on_adapter(const string& name, ValueAccessor<string>& adapter)
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
