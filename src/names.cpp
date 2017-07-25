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

#include <sstream>

#include "names.hpp"

size_t                               NameableValue::__counter = 0;
std::map<std::string, NameableValue> NameableValue::__all_names;

NameableValue::NameableValue(const std::string& name,
                             const std::string& graph_label_type,
                             const std::string& doc_string)
    : m_name{name}
    , m_doc_string{doc_string}
{
    auto glt = m_name;
    if (graph_label_type.size() > 0)
    {
        glt = graph_label_type;
    }

    {
        std::stringstream ss;
        ss << glt << "[" << m_name << "]";
        m_graph_label = ss.str();
    }
}

const std::string& NameableValue::graph_label()
{
    return m_graph_label;
}

const std::string& NameableValue::name()
{
    return m_name;
}

void NameableValue::name(const std::string& name)
{
    // if name == type(self).__name__ or name in NameableValue.__all_names:
    //     while True:
    //         c_name = "{}_{}".format(name, type(self).__counter)
    //         if c_name not in NameableValue.__all_names:
    //             name = c_name
    //             break
    //         type(self).__counter += 1
    // NameableValue.__all_names[name] = self
    // self.__name = name
}

const std::string& NameableValue::short_name()
{
    // sn = self.name.split('_')[0]
    // if sn.find('.') != -1:
    //     sn = sn.split('.')[1]
    // return sn
    static const std::string x = "unimplemented";
    return x;
}

NameableValue& NameableValue::named(const std::string& name)
{
    m_name = name;
    return *this;
}
