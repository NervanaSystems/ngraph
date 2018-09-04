//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
    return m_ss.str();
}

void codegen::CodeWriter::operator+=(const std::string& s)
{
    *this << s;
}

std::string codegen::CodeWriter::generate_temporary_name(std::string prefix)
{
    std::stringstream ss;

    ss << prefix << m_temporary_name_count;
    m_temporary_name_count++;

    return ss.str();
}
