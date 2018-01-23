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

#include "code_writer.hpp"

using namespace std;
using namespace ngraph;

codegen::CodeWriter::CodeWriter()
    : indent(0)
    , m_pending_indent(true)
    , m_temporary_name_count(0)
{
}

string codegen::CodeWriter::get_code() const
{
    stringstream ss;

    for (const string& s : m_include_directives)
    {
        ss << s << "\n";
    }
    ss << m_ss.str();

    return ss.str();
}

stringstream& codegen::CodeWriter::get_stream()
{
    return m_ss;
}

void codegen::CodeWriter::operator+=(const std::string& s)
{
    *this << s;
}

std::string codegen::CodeWriter::generate_temporary_name(std::string prefix)
{
    std::stringstream ss;

    ss << prefix << "__" << m_temporary_name_count;
    m_temporary_name_count++;

    return ss.str();
}

void codegen::CodeWriter::add_include_directive(const std::string& s)
{
    m_include_directives.insert(s);
}
